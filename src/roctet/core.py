import polars as pl
from typing import Literal
from roctet.curves import CurvePiecewise, CurveBeta
from roctet.scores import calc_scores_from_roc

CurveType = Literal["beta", "piecewise"]


def calc_roctet(
    auroc: float,
    method: CurveType = "piecewise",
    n_sets: int = 4,
    n_obsv: int = 100_000,
    event_rate: float = 0.5,
) -> list[pl.DataFrame]:
    """For a given AUC, returns specified number of prediction sets with
    distinct ROC curve patterns but similar AUC values.

    Args:
        auroc (float): Targetted AUROC value for all datasets
        method (CurveType, optional): Method used to derive ROC curve. One of "piecewise" (default) or "beta.
        n_sets (int, optional): Number of datasets (score-target combinations to produce). Defaults to 4.
        n_obsv (int, optional): Number of observations per dataset. Defaults to 1e5.
        event_rate (float, optional): Proportion of positive cases in dataset. Defaults to 0.5.

    Returns:
        list[pl.DataFrame]: One dataset per ROC curve, each approximating the same AUC
    """

    n_neg = int(n_obsv // 2)
    n_pos = n_obsv - n_neg
    curve_classes = {"piecewise": CurvePiecewise, "beta": CurveBeta}
    curve_cls = curve_classes.get(method)
    obj = curve_cls(auroc)
    dfs_roc = obj.gen_rocs(1000, n_sets)
    dfs_sc = [calc_scores_from_roc(d, n_neg, n_pos) for d in dfs_roc]

    return dfs_sc
