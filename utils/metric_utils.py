from sklearn.metrics import confusion_matrix
from numpy import ndarray
from typing import Union


def print_accuracy(tp: int, tn: int, fp: int, fn: int) -> None:
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print("Accuracy: ", accuracy)


def print_recall(tp: int, fn: int) -> None:
    recall = tp / (tp + fn)

    print("Recall: ", recall)


def print_false_positive_rate(fp: int, tn: int) -> None:
    false_positive_rate = fp / (fp + tn)

    print("False Positive Rate: ", false_positive_rate)


def print_false_negative_rate(fn: int, tp: int) -> None:
    false_negative_rate = fn / (fn + tp)

    print("False Negative Rate: ", false_negative_rate)


def print_important_metrics(tp: int, tn: int, fp: int, fn: int) -> None:
    print_accuracy(tp, tn, fp, fn)
    print_recall(tp, fn)
    print_false_positive_rate(fp, tn)
    print_false_negative_rate(fn, tp)


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
