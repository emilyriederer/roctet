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
    - implement `get_control_range(auroc)` (staticmethod) which returns the
      allowable range for the `control` parameter given a target `auroc`.
    - implement `derive_params()` to compute the internal parameters that
      fully describe the underlying shape for the chosen parameterization.
    - implement `gen_roc(n_bin)` to return `n_bin` points on the ROC curve
      as a `polars.DataFrame` with `fpr` and `tpr` columns.

    Args:
        auroc (float): Target area under the ROC curve. Must be in [0, 1].
        control (float): Shape control parameter. Valid range depends on the
            concrete subclass and should be validated by calling
            `get_control_range(auroc)`.
    """

    def __init__(self, auroc: float) -> None:
        if auroc > 1 or auroc < 0:
            raise ValueError(f"Invalid `auroc` ({auroc}). Must be in (0,1).")
        self.auroc = auroc
        range = self.get_control_range()
        self.control = (range[0]+range[1]) / 2

    @property
    @abstractmethod
    def get_control_range() -> tuple[float, float]:
        """Return (min, max) allowable `control` values for the given `auroc`."""

    @staticmethod
    @abstractmethod
    def get_control_transform() -> Callable[[str], bool]:
        """Returns function that transforms the range of the control."""

    @abstractmethod
    def derive_params(self, control: float = None) -> dict[str, float]:
        """Derive internal parameters from `auroc` and `control`.

        Returns a tuple of floats representing the parameters that uniquely
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

        control_rng = self.get_control_range()
        control_trn = self.__class__.get_control_transform()
        controls = [control_trn(x) for x in np.linspace(control_rng[0],control_rng[1],n_sets)]
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
        super().__init__(auroc)

    def get_control_range(auroc: float | None = None) -> tuple[float, float]:
        """Return allowed range for `control` for the beta parameterization.

        This is a simple heuristic range used by the original implementation.
        """
        return (-1,6)

    def get_control_transform() -> Callable[[float], float]:
        """Returns function that transforms the range of the control."""
        return lambda x: 2**x

    def derive_params(self, control: float = None) -> dict[str, float]:
        """Derive Beta distribution parameters `(a, b)`.

        Returns:
            tuple[float, float]: `(alpha, beta)` parameters for the Beta CDF.
        """
        control = control or self.control
        b = self.auroc * control
        a = control - b
        return {'a': a, 'b': b}

    def gen_roc(self, n_bin: int, control: float = None) -> pl.DataFrame:
        """Create dataset with points on the ROC curve (Beta CDF).

        Args:
            n_bin (int): Number of points to output. Must be >= 1.

        Raises:
            ValueError: If `n_bin < 1`.

        Returns:
            polars.DataFrame: DataFrame with columns `fpr` and `tpr`.
        """
        control = control or self.control
        params = self.derive_params(control)
        a,b = params.values()
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
        super().__init__(auroc)

    def get_control_range(self) -> tuple[float, float]:
        """Return allowable `control` range for the piecewise parameterization.

        The heuristic enforces a small epsilon away from 0 and 1 and ensures the
        trapezoid can achieve the requested `auroc`.
        """
        control_min = max(0.01,1-2*self.auroc)
        control_max = min(0.99,2-2*self.auroc)
        return (control_min, control_max)

    def get_control_transform() -> Callable[[float], float]:
        """Returns function that transforms the range of the control."""
        return lambda x: x

    def derive_params(self, control: float = None) -> dict[str, float]:
        """Derive the inflection point `(x, y)` for the trapezoid curve.

        Returns:
            tuple[float, float]: `(x, y)` where `x` is the inflection x-coordinate
            (distance along the x axis) and `y` is the corresponding height.
        """
        control = control or self.control
        x = control
        y = 2*self.auroc + x - 1
        return {'x': x, 'y': y}

    def gen_roc(self, n_bin: int, control: float = None) -> pl.DataFrame:
        """Create dataset with points on the ROC curve (linear interpolation).

        Args:
            n_bin (int): Number of points to output. Must be >= 1.

        Raises:
            ValueError: If `a` or `b` are out of [0,1] or if `n_bin < 1`.

        Returns:
            polars.DataFrame: DataFrame with `fpr` and `tpr`.
        """
        control = control or self.control
        params = self.derive_params(control)
        x,y = params.values()
        if n_bin < 1:
            raise ValueError(f"Invalid `n_bin` of {n_bin}. Must be at least 1.")
        fpr = np.linspace(0, 1, n_bin)
        tpr = np.where(fpr < x, (y/x)*fpr, y+(1-y)*(fpr-x)/(1-x)) 
        df_roc = pl.DataFrame({"fpr": fpr, "tpr": tpr})
        return df_roc