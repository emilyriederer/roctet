import polars as pl
import numpy as np
from numpy.random import default_rng


def _gen_roc_to_scorebins(df_roc: pl.DataFrame, n_neg: int, n_pos: int) -> pl.DataFrame:

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


def _gen_scorebins_to_scores(
    df_scorebins: pl.DataFrame, seed: int = 123
) -> pl.DataFrame:

    # TODO: validation df_scorebins, seeds

    def _gen_scores_arr(n_pos: int, n_neg: int) -> np.ndarray:

        more = 1 if n_pos >= n_neg else 0
        xtra = np.repeat(more, abs(n_pos - n_neg))
        base = np.tile([0, 1], min(n_pos, n_neg))
        out = np.append(base, xtra)
        return out

    rng = default_rng(seed)

    df_scores = df_scorebins.with_columns(
        score=pl.struct("score_min", "score_max", "n").map_elements(
            function=lambda z: sorted(
                rng.uniform(
                    low=z["score_min"], high=z["score_max"], size=z["n"]
                ).tolist()
            ),
            return_dtype=pl.List(pl.Float64),
        ),
        target=pl.struct("n", "n_pos").map_elements(
            function=lambda z: rng.binomial(
                n=1, p=z["n_pos"] / z["n"], size=z["n"]
            ).tolist(),
            return_dtype=pl.List(pl.Int64),
        ),
        # target = pl.struct('n_neg','n_pos').map_elements(
        #    function = lambda d: _gen_scores_arr(d['n_pos'],d['n_neg']),
        #    return_dtype = pl.List(pl.Int64))
    )
    return df_scores.explode("score", "target")
