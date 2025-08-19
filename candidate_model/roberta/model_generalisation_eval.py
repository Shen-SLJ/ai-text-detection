from candidate_model.roberta.model import CandidateRobertaModel
from dataset_processing.datasets import get_eval_dataset
from utils.metric_utils import print_important_metrics, get_confusion_matrix_as_tuple


X_eval, y_eval = get_eval_dataset()

roberta_model = CandidateRobertaModel(load_from_saved=True)

prediction = roberta_model.predict(X_eval)

tn, fp, fn, tp = get_confusion_matrix_as_tuple(y_eval, prediction)

print("Evaluation results: ")
print_important_metrics(tp, tn, fp, fn)
