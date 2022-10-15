# %% Import packages
from tkinter.tix import Y_REGION
import numpy as np
from sklearn.metrics import (confusion_matrix,
                             precision_recall_curve)
import utils


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
m, n = np.shape(y_true)
conf = [0, 0, 0, 0]
for i in range(m):
    for j in range(n):
        conf[cm[i][j]] += 1
tnc, tpc, fnc, fpc = conf
print("tp, fn")
print("fp, tn")
print(tpc, fnc)
print(fpc, tnc)
print("accuracy:", (tpc + tnc) / (tpc + tnc + fpc + fnc))
precision = tpc / (tpc + fpc)
print("precision:", precision)
recall = tpc / (tpc + fnc)
print("recall:", recall)
print("F1-Score:", 2 * recall * precision / (recall + precision))

classes = [0, 0, 0, 0, 0, 0, 0]
for i in range(m):
    classes[y_true[i].argmax()] += 1
print(classes)
y_true = np.array(utils.get_all_hea("./data/"))
m, n = np.shape(y_true)
classes = [0, 0, 0, 0, 0, 0, 0]
for i in range(m):
    classes[y_true[i].argmax()] += 1
print(classes)
sum = sum(classes)
print(sum)
for i in classes:
    print(1-i/sum)
