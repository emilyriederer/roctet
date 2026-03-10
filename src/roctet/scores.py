import polars as pl
import numpy as np
from numpy.random import default_rng

def _gen_roc_to_scorebins(df_roc: pl.DataFrame, n_neg: int, n_pos: int) -> pl.DataFrame:
    """Derive score bins and frequencies of true positives and true negatives

    Args:
        df_roc (pl.DataFrame): ROC curve characterized as dataset with (`fpr`,`tpr`) pairs
        n_neg (int): Global number of true negatives in target dataset
        n_pos (int): Global number of true positives in target dataset

    Returns:
        pl.DataFrame: Dataset containing (`score_min`,`score_max`,`n_pos`,`n_neg`,`n`)
    """

    # TODO: Validate df_roc

    # derive score bin to observation bin mapping
    df_scorebins = (
        df_roc.sort("fpr")
        .with_columns(
            score_max=pl.col("fpr").rank("ordinal", descending=True)
            / (pl.col("fpr").len() - 1),
            cum_n_neg=pl.col("fpr").mul(n_neg).round(0).cast(pl.Int64),
            cum_n_pos=pl.col("tpr").mul(n_pos).round(0).cast(pl.Int64),
        )
        .with_columns(
            score_min=pl.col("score_max").shift(-1).fill_null(0),
            n_neg=pl.col("cum_n_neg") - pl.col("cum_n_neg").shift(1).fill_null(0),
            n_pos=pl.col("cum_n_pos") - pl.col("cum_n_pos").shift(1).fill_null(0),
        )
        .filter(pl.col("score_max").is_between(0, 1))
        .with_columns(n=pl.col("n_neg") + pl.col("n_pos"))
        .select("score_min", "score_max", "n_pos", "n_neg", "n")
    )

    return df_scorebins


def _gen_scorebins_to_scores(df_scorebins: pl.DataFrame, seed: int = 123) -> pl.DataFrame:
    """Generate score-level dataset from scorebins

    Args:
        df_scorebins (pl.DataFrame): Scorebin data as returned by `_gen_roc_to_scorebins()`
        seed (int, optional): Random seed for binomial and uniform sampling. Defaults to 123.

    Returns:
        pl.DataFrame: Dataset containing `score` and `target` for each sample observation
    """

    # add a stable row index for per-row RNG seed
    # this avoids nondeterminism when `polars` executes `map_elements` out of order
    df_scorebins = df_scorebins.with_row_index("_row_idx")

    def _score_fn(z):
        r = default_rng(int(seed) + int(z["_row_idx"]))
        return r.uniform(low=z["score_min"], high=z["score_max"], size=z["n"]).tolist()

    def _target_fn(z):
        r = default_rng(int(seed) + int(z["_row_idx"]) + 1)
        return r.binomial(n=1, p=z["n_pos"] / z["n"], size=z["n"]).tolist()

    df_scores = df_scorebins.with_columns(
        score=pl.struct("score_min", "score_max", "n", "_row_idx").map_elements(
            function=_score_fn,
            return_dtype=pl.List(pl.Float64),
        ),
        target=pl.struct("n", "n_pos", "_row_idx").map_elements(
            function=_target_fn,
            return_dtype=pl.List(pl.Int64),
        ),
    )
    return df_scores.explode("score", "target")
