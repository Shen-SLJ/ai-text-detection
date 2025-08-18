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