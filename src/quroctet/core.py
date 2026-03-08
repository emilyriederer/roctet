import polars as pl
import numpy as np
from quroctet import curves, scores

def calc_roctet(auc:float, n_sets:int=4, n_obsv:int=1e5) -> list[pl.DataFrame]:
    """For a given AUC, returns specified number of prediction sets with
    distinct ROC curve patterns but similar AUC values.

    Args:
        auc (float): Targetted AUC value for all datasets
        n_sets (int, optional): Number of datasets (score-target combinations to produce). Defaults to 4.
        n_obsv (int, optional): Number of observations per dataset. Defaults to 1e5.

    Returns:
        list[pl.DataFrame]: One dataset per ROC curve, each approximating the same AUC
    """

    sum_controls = 2**np.linspace(-1,6,n_sets)
    n_neg = np.floor( n_obsv * 0.5 )
    n_pos = n_obsv - n_neg

    params = [curves._derive_params(auc, s) for s in sum_controls]
    dfs_roc = [curves._gen_roc(p[0],p[1],1000) for p in params]
    dfs_sb = [scores._gen_roc_to_scorebins(d, n_neg, n_pos) for d in dfs_roc]
    dfs_sc = [scores._gen_scorebins_to_scores(d) for d in dfs_sb]
    return dfs_sc

if __name__ == "__main__":
    from utils import auc
    [np.round( auc(d['target'].to_numpy(), d['score'].to_numpy()),3) for d in calc_roctet(0.3)]
    breakpoint()