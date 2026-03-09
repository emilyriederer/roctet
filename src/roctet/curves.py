import numpy as np
import polars as pl
from scipy.stats import beta


def _derive_params(auc: float, sum_control: float) -> tuple[float, float]:
    """Derive beta parameters for AUC curve

    Derive alpha and beta parameters for a Beta distribution based on the targeted AUC
    and a measure of spread/certainty. AUC under the CDF is essentially the expected
    value of the Beta distribution (b / b+a), and the sum of alpha and beta characterizes
    the certainty of the distribution.

    Mathematically, the beta distribution is unrelated to AUROC in the machine learning
    context. It is used here as a simple, closed-form function with attractive properties
    for generating an AUC curve, such as being continuous, monotonic, bounded between 0 and 1,
    and easily parameterized.

    Args:
        auc (float): Target area under the curve (expected value of Beta distribution)
        sum_control (float): Target sum of alpha and beta parameters (certainty)

    Returns:
        tuple[float,float]: Contains values of alpha and beta
    """

    if auc > 1 or auc < 0:
        raise ValueError(f"Invalid AUC of {auc} provided. Must be between 0 and 1.")
    if sum_control < 0:
        raise ValueError(
            f"Invalid `sum_control` of {sum_control} provided. Must be positive."
        )

    b = auc * sum_control
    a = sum_control - b
    return (a, b)


def _gen_roc(a: float, b: float, n_bin: int = 25) -> pl.DataFrame:
    """Create dataset with points on the ROC curve (parameterized as Beta CDF)

    Args:
        a (float): Alpha parameter of Beta distribution
        b (float): Beta parameter of the Beta distribution
        n_bin (int, optional): Number of points to output. Defaults to 25.

    Raises:
        ValueError: Negative values for alpha or beta parameters
        ValueError: Insufficient (<1) values for bins to be produced

    Returns:
        pl.DataFrame: Dataframe with points representing ROC curve as (`fpr`,`tpr`)
    """

    if a < 0 or b < 0:
        raise ValueError(f"Invalid Beta distributino parameters ({a},{b}). Must be positive.")
    if n_bin < 1:
        raise ValueError(f"Invalid `n_bin` of {n_bin}. Must be at least 1.")
    fpr = np.linspace(0, 1, n_bin)
    tpr = beta(a, b).cdf(fpr)
    df_roc = pl.DataFrame({"fpr": fpr, "tpr": tpr})
    return df_roc
