from sklearn.metrics import confusion_matrix
from numpy import ndarray
from typing import Union


def get_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    return (tp + tn) / (tp + tn + fp + fn)


def get_recall(tp: int, fn: int) -> float:
    return tp / (tp + fn)


def get_false_positive_rate(fp: int, tn: int) -> float:
    return fp / (fp + tn)


def get_false_negative_rate(fn: int, tp: int) -> float:
    return fn / (fn + tp)


def print_important_metrics(tp: int, tn: int, fp: int, fn: int) -> float:
    print("Accuracy: ", get_accuracy(tp, tn, fp, fn))
    print("Recall: ", get_recall(tp, fn))
    print("False positive rate: ", get_false_positive_rate(fp, tn))
    print("False negative rate: ", get_false_negative_rate(fn, tp))


def get_confusion_matrix_as_tuple(
    true_labels: Union[ndarray, list], pred_labels: Union[ndarray, list]
) -> tuple[int, int, int, int]:
    """
    Returns the confusion matrix as a tuple (TN, FP, FN, TP).

    Returns:
      tuple: A tuple containing (TN, FP, FN, TP).
    """
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel().tolist()

    return tn, fp, fn, tp
