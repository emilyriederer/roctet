import polars as pl
import numpy as np
from roctet.scores import (
    _gen_roc_to_scorebins,
    _gen_scorebins_to_scores,
)

def _make_simple_scorebins():
    # create a small ROC that maps evenly into bins
    df_roc = pl.DataFrame({
        "fpr": [0.0, 0.5, 1.0],
        "tpr": [0.0, 0.5, 1.0],
    })
    return _gen_roc_to_scorebins(df_roc, n_neg=4, n_pos=6)


def test_reproducible_with_default_rng():

    df_bins = _make_simple_scorebins()

    df_a = _gen_scorebins_to_scores(df_bins, seed=123)
    df_b = _gen_scorebins_to_scores(df_bins, seed=123)

    arr_a = df_a['score'].to_numpy()
    arr_b = df_b['score'].to_numpy()
    assert np.allclose(arr_a, arr_b)


def test_different_rng_seeds_change_output():

    df_bins = _make_simple_scorebins()

    df_a = _gen_scorebins_to_scores(df_bins, seed=123)
    df_b = _gen_scorebins_to_scores(df_bins, seed=456)

    arr_a = df_a['score'].to_numpy()
    arr_b = df_b['score'].to_numpy()
    assert not np.allclose(arr_a, arr_b)
