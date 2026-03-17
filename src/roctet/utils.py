import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata


def auroc(y_true: npt.NDArray[np.floating], y_pred: npt.NDArray[np.floating]) -> float:
    """Compute AUC using Mann-Whitney U Statistics

    Args:
        y_true (npt.NDArray[np.floating]): Binary 0/1 indicators of true value
        y_pred (npt.NDArray[np.floating]): Scores predicting the binary truth

    Returns:
        float: AUROC statistics
    """
    # rank all predictions
    ranks = rankdata(y_pred)

    # add rank numbers where target is positive
    rank_sum_pos = np.sum(ranks[y_true == 1])

    # calc stats on number of negative and positive
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos

    # apply Mann-Whitney U statistics
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    return round(auc.item(), 4)
