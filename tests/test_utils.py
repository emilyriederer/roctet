import numpy as np
from roctet.utils import auroc

def test_auroc_basic():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    assert auroc(y_true, y_pred) == 0.75

def test_auroc_ties():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.5, 0.5, 0.5, 0.5])
    assert auroc(y_true, y_pred) == 0.5
