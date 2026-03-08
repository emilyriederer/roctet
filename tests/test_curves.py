import numpy as np
import polars as pl
from roctet.curves import _derive_params, _gen_roc


def test_derive_params_sum_and_ratio():
    auc = 0.7
    sum_control = 10.0
    a, b = _derive_params(auc, sum_control)
    assert np.isclose(a + b, sum_control)
    assert np.isclose(b / (a + b), auc)


def test_gen_roc_properties():
    a, b = _derive_params(0.6, 5.0)
    df = _gen_roc(a, b, n_bin=11)
    assert isinstance(df, pl.DataFrame)
    assert df.height == 11
    assert set(df.columns) == {"fpr", "tpr"}
    # monotonic non-decreasing tpr
    tpr = df['tpr'].to_numpy()
    assert np.all(np.diff(tpr) >= -1e-12)
    # values in [0,1]
    assert np.all(tpr >= 0) and np.all(tpr <= 1)
