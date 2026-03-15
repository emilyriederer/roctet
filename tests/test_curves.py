import numpy as np
import polars as pl
import pytest

from roctet.curves import CurveBeta, CurvePiecewise


@pytest.mark.parametrize("curve_cls", [CurveBeta, CurvePiecewise])
@pytest.mark.parametrize("auroc", [0.3, 0.5, 0.8])
def test_constructor_valid_auroc(curve_cls, auroc):
    curve = curve_cls(auroc)
    assert curve.auroc == auroc
    control_min, control_max = curve.control_range
    assert control_min < control_max
    assert control_min <= curve.control <= control_max


@pytest.mark.parametrize("curve_cls", [CurveBeta, CurvePiecewise])
@pytest.mark.parametrize("auroc", [0.0, 1.0, -0.1, 1.1])
def test_constructor_invalid_auroc_raises(curve_cls, auroc):
    with pytest.raises(ValueError):
        curve_cls(auroc)


@pytest.mark.parametrize("curve_cls", [CurveBeta, CurvePiecewise])
def test_gen_roc_properties(curve_cls):
    curve = curve_cls(0.6)
    control_min, control_max = curve.control_range
    control = (control_min + control_max) / 2
    n_bin = 11

    df = curve.gen_roc(n_bin=n_bin, control=control)

    assert isinstance(df, pl.DataFrame)
    assert df.height == n_bin
    assert set(df.columns) == {"fpr", "tpr"}

    fpr = df["fpr"].to_numpy()
    tpr = df["tpr"].to_numpy()

    # fpr spans [0, 1] and is non-decreasing
    assert np.isclose(fpr[0], 0.0)
    assert np.isclose(fpr[-1], 1.0)
    assert np.all(np.diff(fpr) >= 0)

    # tpr is non-decreasing and stays in [0, 1]
    assert np.all(np.diff(tpr) >= -1e-12)
    assert np.all(tpr >= -1e-12) and np.all(tpr <= 1 + 1e-12)


@pytest.mark.parametrize("curve_cls", [CurveBeta, CurvePiecewise])
def test_gen_roc_invalid_n_bin_raises(curve_cls):
    curve = curve_cls(0.6)
    control_min, control_max = curve.control_range
    control = (control_min + control_max) / 2

    for bad_n_bin in [0, -1]:
        with pytest.raises(ValueError):
            curve.gen_roc(n_bin=bad_n_bin, control=control)


@pytest.mark.parametrize("curve_cls", [CurveBeta, CurvePiecewise])
def test_gen_rocs_generates_multiple_curves(curve_cls):
    curve = curve_cls(0.7)
    n_bin = 21
    n_sets = 5

    curves = curve.gen_rocs(n_bin=n_bin, n_sets=n_sets)
    assert len(curves) == n_sets

    for df in curves:
        assert isinstance(df, pl.DataFrame)
        assert df.height == n_bin
        assert set(df.columns) == {"fpr", "tpr"}


def test_curve_beta_derive_params_sum_and_ratio():
    auc = 0.7
    sum_control = 10.0
    curve = CurveBeta(auc)
    params = curve.derive_params(control=sum_control)
    a, b = params["a"], params["b"]

    assert np.isclose(a + b, sum_control)
    assert np.isclose(b / (a + b), auc)


def test_curve_piecewise_derive_params_relationship():
    auc = 0.7
    curve = CurvePiecewise(auc)
    control_min, control_max = curve.control_range
    control = (control_min + control_max) / 2

    params = curve.derive_params(control=control)
    x, y = params["x"], params["y"]

    assert np.isclose(x, control)
    assert np.isclose(y, 2 * auc + x - 1)

