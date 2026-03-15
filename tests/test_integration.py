import polars as pl
from roctet.core import calc_roctet
from roctet.utils import auroc


def test_calc_roctet_integration():

    target_auc = 0.7
    dfs = calc_roctet(target_auc, n_sets=2, n_obsv=2000)
    assert isinstance(dfs, list)
    assert len(dfs) == 2
    for d in dfs:
        assert isinstance(d, pl.DataFrame)
        assert set(["score", "target"]).issubset(set(d.columns))
        computed = auroc(d["target"].to_numpy(), d["score"].to_numpy())
        assert abs(computed - target_auc) < 0.05
