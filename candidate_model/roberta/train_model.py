from candidate_model.roberta.model import CandidateRobertaModel
from dataset_processing.datasets import get_train_dataset
from sklearn.model_selection import train_test_split
from utils.metric_utils import print_important_metrics, get_confusion_matrix_as_tuple


documents, labels = get_train_dataset()

X_train, X_temp, y_train, y_temp = train_test_split(
    documents, labels, test_size=0.2, random_state=0
)
X_eval, X_test, y_eval, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=0
)
X_train, X_test, X_eval, y_train, y_test, y_eval = (
    X_train.tolist(),
    X_test.tolist(),
    X_eval.tolist(),
    y_train.tolist(),
    y_test.tolist(),
    y_eval.tolist(),
)

candidate_model = CandidateRobertaModel(X_train, y_train, X_eval, y_eval).train()

prediction = candidate_model.predict(X_test)

tn, fp, fn, tp = get_confusion_matrix_as_tuple(y_test, prediction)

print("Training evaluation results: ")
print_important_metrics(tp, tn, fp, fn)
