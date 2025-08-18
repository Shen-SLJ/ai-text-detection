from sklearn.metrics import confusion_matrix
from numpy import ndarray


def get_confusion_matrix_as_tuple(true_labels: ndarray, pred_labels: ndarray) -> tuple[int, int, int, int]:
  """
  Returns the confusion matrix as a tuple (TN, FP, FN, TP).

  Returns:
    tuple: A tuple containing (TN, FP, FN, TP).
  """
  tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel().tolist()

  return tn, fp, fn, tp