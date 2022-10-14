# %% Import packages
from tkinter.tix import Y_REGION
import numpy as np
from sklearn.metrics import (confusion_matrix,
                             precision_recall_curve)
import utils


def get_optimal_precision_recall(y_true, y_score):
    """Find precision and recall values that maximize f1 score."""
    n = np.shape(y_true)[1]
    opt_precision = []
    opt_recall = []
    opt_threshold = []
    for k in range(n):
        # Get precision-recall curve
        precision, recall, threshold = precision_recall_curve(y_true[:, k], y_score[:, k])
        # Compute f1 score for each point (use nan_to_num to avoid nans messing up the results)
        f1_score = np.nan_to_num(2 * precision * recall / (precision + recall))
        # Select threshold that maximize f1 score
        index = np.argmax(f1_score)
        opt_precision.append(precision[index])
        opt_recall.append(recall[index])
        t = threshold[index - 1] if index != 0 else threshold[0] - 1e-10
        opt_threshold.append(t)
    return np.array(opt_precision), np.array(opt_recall), np.array(opt_threshold)


def affer_results(y_true, y_pred):
    """Return true positives, false positives, true negatives, false negatives.

    Parameters
    ----------
    y_true : ndarray
        True value
    y_pred : ndarray
        Predicted value

    Returns
    -------
    tn, tp, fn, fp: ndarray
        Boolean matrices containing true negatives, true positives, false negatives and false positives.
    cm : ndarray
        Matrix containing: 0 - true negative, 1 - true positive,
        2 - false negative, and 3 - false positive.
    """
    # True negative
    tn = (y_true == y_pred) & (y_pred == 0)
    # True positive
    tp = (y_true == y_pred) & (y_pred == 1)
    # False positive
    fp = (y_true != y_pred) & (y_pred == 1)
    # False negative
    fn = (y_true != y_pred) & (y_pred == 0)
    # Generate matrix of "tp, fp, tn, fn"
    m, n = np.shape(y_true)
    cm = np.zeros((m, n), dtype=int)
    cm[tn] = 0
    cm[tp] = 1
    cm[fn] = 2
    cm[fp] = 3
    return tn, tp, fn, fp, cm


def nomalize(y_pred):
    old = y_pred
    m, n = np.shape(y_pred)
    y_pred = np.zeros((m, n), dtype=int)
    for i in range(m):
        y_pred[i][old[i].argmax()] = 1
    return y_pred


# Get true values
y_true = np.array(utils.get_all_hea("./data/test_set/"))
y_pred = nomalize(np.load('./dnn_output.npy'))
tn, tp, fn, fp, cm = affer_results(y_true, y_pred)
print("tn, tp, fn, fp")
