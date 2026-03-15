from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
import polars as pl
from scipy.stats import beta


class CurveShaper(ABC):
    """
    Abstract base class for ROC-curve shapers.

    Subclasses implement a parameterization that maps a target AUROC value
    and a `control` parameter (which encodes spread, inflection, or other
    shape control) to a pair of derived parameters and a generated ROC
    curve as a `polars.DataFrame` with columns `fpr` and `tpr`.

    Responsibilities of subclasses:
    - implement `control_range()` (property) which returns the
      allowable range for the `control` parameter given a target `auroc`.
    - implement `derive_params()` to compute the internal parameters that
      fully describe the underlying shape for the chosen parameterization.
    - implement `gen_roc(n_bin)` to return `n_bin` points on the ROC curve
      as a `polars.DataFrame` with `fpr` and `tpr` columns.

    Args:
        auroc (float): Target area under the ROC curve. Must be in (0, 1).
        control_transform (Callable[[float], float]): Function to transform control parameter.
    """

    def __init__(
        self, auroc: float, control_transform: Callable[[float], float]
    ) -> None:
        if auroc >= 1 or auroc <= 0:
            raise ValueError(f"Invalid `auroc` ({auroc}). Must be in (0,1).")
        self.auroc = auroc
        self.control_transform = control_transform
        control_range = self.control_range
        self.control = (control_range[0] + control_range[1]) / 2

    def _validate_control(self, control: float) -> float:
        """Internal helper to confirm user-provided control or revert to class default

        Args:
            control (float): User-provided input value for control

        Raises:
            ValueError: Triggered on invalid control values based on AUC and curve-imposed constraints

        Returns:
            float: The control value to use for downstream processing
        """
        if control is None:
            return self.control
        control_range = self.control_range
        if control < control_range[0] or control > control_range[1]:
            raise ValueError(
                f"Invalid `control` ({control}). Must be in {control_range}"
            )
        return control

    @property
    @abstractmethod
    def control_gen_range(self) -> tuple[float, float]:
        """Return (min, max) allowable pre-transformation `control` values for the given `auroc`."""

    @property
    def control_range(self) -> tuple[float, float]:
        """Return (min, max) allowable `control` values for the given `auroc`."""
        crg = self.control_gen_range
        return (self.control_transform(crg[0]), self.control_transform(crg[1]))

    @abstractmethod
    def derive_params(self, control: float = None) -> dict[str, float]:
        """Derive internal parameters from `auroc` and `control`.

        Returns a dict of floats representing the parameters that uniquely
        specify the curve shape for the concrete parameterization.
        """

    @abstractmethod
    def gen_roc(self, n_bin: int, control: float) -> pl.DataFrame:
        """Generate `n_bin` points on the ROC curve as a `polars.DataFrame`.

        The returned DataFrame must have columns `fpr` and `tpr` and contain
        monotonically increasing values from (0,0) to (1,1) (subject to the
        concrete parameterization).
        """

    def gen_rocs(self, n_bin: int, n_sets: int) -> list[pl.DataFrame]:

        cgr = self.control_gen_range
        control_trn = self.control_transform
        controls = [control_trn(x) for x in np.linspace(cgr[0], cgr[1], n_sets)]
        curves = [self.gen_roc(n_bin, c) for c in controls]
        return curves


class CurveBeta(CurveShaper):
    """
    ROC curve parameterization using the Beta CDF.

    This class mirrors the former `curves.py` implementation. It models the
    ROC curve as the CDF of a Beta distribution parameterized by
    `alpha` and `beta`. The target `auroc` is treated as the expected value
    of that Beta distribution (b/(a+b)), and the `control` value is the sum
    `a+b` which acts as a concentration/certainty parameter.

    The class exposes `derive_params()` which returns `(a, b)` and
    `gen_roc(n_bin)` which returns a `polars.DataFrame` with `n_bin` points on
    the curve.
    """

    def __init__(self, auroc: float) -> None:
        super().__init__(auroc, lambda x: 2**x)

    @property
    def control_gen_range(self) -> tuple[float, float]:
        """Return allowed range for `control` for the beta parameterization.

        This is a simple heuristic range used by the original implementation.
        """
        return (-1, 6)

    def derive_params(self, control: float = None) -> dict[str, float]:
        """Derive Beta distribution parameters `(a, b)`.

        Returns:
            dict[str, float]: `(alpha, beta)` parameters for the Beta CDF.
        """
        control = self._validate_control(control)
        b = self.auroc * control
        a = control - b
        return {"a": a, "b": b}

    def gen_roc(self, n_bin: int, control: float = None) -> pl.DataFrame:
        """Create dataset with points on the ROC curve (Beta CDF).

        Args:
            n_bin (int): Number of points to output. Must be >= 1.

        Raises:
            ValueError: If `n_bin < 1`.

        Returns:
            polars.DataFrame: DataFrame with columns `fpr` and `tpr`.
        """
        control = self._validate_control(control)
        params = self.derive_params(control)
        a, b = params.values()
        if n_bin < 1:
            raise ValueError(f"Invalid `n_bin` of {n_bin}. Must be at least 1.")
        fpr = np.linspace(0, 1, n_bin)
        tpr = beta(a, b).cdf(fpr)
        df_roc = pl.DataFrame({"fpr": fpr, "tpr": tpr})
        return df_roc


class CurvePiecewise(CurveShaper):
    """
    Piecewise linear (trapezoid) ROC curve parameterization.

    This class mirrors the former `curves_interp.py` implementation. The curve
    is parameterized as a four-sided trapezoid composed of a triangle followed
    by a rectangle+triangle so that the overall shape sits on the x-axis and
    connects (0,0) to (1,1). The `control` parameter represents the x-axis
    location of the inflection point; `derive_params()` returns `(x, y)` where
    `x` is the inflection x-coordinate and `y` is the corresponding height.

    `gen_roc(n_bin)` returns linearly interpolated points between the defined
    segments and outputs a `polars.DataFrame` with `fpr` and `tpr`.
    """

    def __init__(self, auroc: float) -> None:
        super().__init__(auroc, lambda x: x)

    @property
    def control_gen_range(self) -> tuple[float, float]:
        """Return allowable `control` range for the piecewise parameterization.

        The heuristic enforces a small epsilon away from 0 and 1 and ensures the
        trapezoid can achieve the requested `auroc`.
        """
        control_min = max(0.01, 1 - 2 * self.auroc)
        control_max = min(0.99, 2 - 2 * self.auroc)
        return (control_min, control_max)

    def derive_params(self, control: float = None) -> dict[str, float]:
        """Derive the inflection point `(x, y)` for the trapezoid curve.

        Returns:
            dict[str, float]: `(x, y)` where `x` is the inflection x-coordinate
            (distance along the x axis) and `y` is the corresponding height.
        """
        control = self._validate_control(control)
        x = control
        y = 2 * self.auroc + x - 1
        return {"x": x, "y": y}

    def gen_roc(self, n_bin: int, control: float = None) -> pl.DataFrame:
        """Create dataset with points on the ROC curve (linear interpolation).

        Args:
            n_bin (int): Number of points to output. Must be >= 1.

        Raises:
            ValueError: If `n_bin < 1`.

        Returns:
            polars.DataFrame: DataFrame with `fpr` and `tpr`.
        """
        control = self._validate_control(control)
        params = self.derive_params(control)
        x, y = params.values()
        if n_bin < 1:
            raise ValueError(f"Invalid `n_bin` of {n_bin}. Must be at least 1.")
        fpr = np.linspace(0, 1, n_bin)
        tpr = np.where(fpr < x, (y / x) * fpr, y + (1 - y) * (fpr - x) / (1 - x))
        df_roc = pl.DataFrame({"fpr": fpr, "tpr": tpr})
        return df_roc
