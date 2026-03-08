import polars as pl
from roctet.scores import _gen_roc_to_scorebins, _gen_scorebins_to_scores

def test_roc_to_scorebins_counts():
    # simple ROC that maps evenly
    df_roc = pl.DataFrame({
        'fpr': [0.0, 0.5, 1.0],
        'tpr': [0.0, 0.5, 1.0]
    })
    n_neg = 4
    n_pos = 6
    df_bins = _gen_roc_to_scorebins(df_roc, n_neg, n_pos)
    # total counts should sum to the provided totals
    assert int(df_bins['n_pos'].sum()) == n_pos
    assert int(df_bins['n_neg'].sum()) == n_neg
    assert int(df_bins['n'].sum()) == (n_pos + n_neg)
    # correct values by row
    assert list(df_bins['n_pos']) == [3,3]
    assert list(df_bins['n_neg']) == [2,2]

def test_roc_to_scorebins_counts_complex():
    # simple ROC that maps evenly
    df_roc = pl.DataFrame({
        'fpr': [0.0, 0.2, 0.5, 0.8, 1],
        'tpr': [0.0, 0.4, 0.75, 0.9, 1]
    })
    n_neg = 100
    n_pos = 100
    df_bins = _gen_roc_to_scorebins(df_roc, n_neg, n_pos)
    # total counts should sum to the provided totals
    assert int(df_bins['n_pos'].sum()) == n_pos
    assert int(df_bins['n_neg'].sum()) == n_neg
    assert int(df_bins['n'].sum()) == (n_pos + n_neg)
    # correct values by row
    assert list(df_bins['n_neg']) == [20,30,30,20]
    assert list(df_bins['n_pos']) == [40,35,15,10]
    assert list(df_bins['score_min']) == [0.75,0.5,0.25,0]
    assert list(df_bins['score_max']) == [1,0.75,0.5,0.25]

def test_scorebins_to_scores_basic():
    df_roc = pl.DataFrame({
        'fpr': [0.0, 0.5, 1.0],
        'tpr': [0.0, 0.5, 1.0]
    })
    n_neg = 2
    n_pos = 2
    df_bins = _gen_roc_to_scorebins(df_roc, n_neg, n_pos)
    df_scores = _gen_scorebins_to_scores(df_bins)
    # exploded frame should have n_pos + n_neg rows
    assert df_scores.height == n_pos + n_neg

