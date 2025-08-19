from candidate_model.roberta.model import CandidateRobertaModel
from dataset_processing.datasets import get_train_dataset
from sklearn.model_selection import train_test_split
from utils.metric_utils import print_important_metrics, get_confusion_matrix_as_tuple


documents, labels = get_train_dataset()

# Temp test code
documents = documents[0:10]
labels = labels[0:10]

X_train, X_test, y_train, y_test = train_test_split(
    documents, labels, test_size=0.2, random_state=0
)
X_train, X_test, y_train, y_test = (
    X_train.tolist(),
    X_test.tolist(),
    y_train.tolist(),
    y_test.tolist(),
)

candidate_model = CandidateRobertaModel(X_train, y_train).train()

prediction = candidate_model.predict(X_test)

tn, fp, fn, tp = get_confusion_matrix_as_tuple(y_test, prediction)

print("Training evaluation results: ")
print_important_metrics(tp, tn, fp, fn)
