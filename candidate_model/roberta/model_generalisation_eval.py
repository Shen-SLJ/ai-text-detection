from candidate_model.roberta.model import RobertaModel
from dataset_processing.datasets import get_eval_dataset
from utils.metric_utils import print_important_metrics, get_confusion_matrix_as_tuple


X_eval, y_eval = get_eval_dataset()
X_eval = X_eval.tolist()

roberta_model = RobertaModel(load_checkpoint_number=5250)

prediction = roberta_model.predict(X_eval)

tn, fp, fn, tp = get_confusion_matrix_as_tuple(y_eval, prediction)

print("Evaluation results: ")
print_important_metrics(tp, tn, fp, fn)
