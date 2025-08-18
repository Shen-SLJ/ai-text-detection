from baseline_model.baseline_model import BaselineModel
from dataset_processing.datasets import get_eval_dataset
from utils.sklearn_utils import get_confusion_matrix_as_tuple
from utils.metric_utils import print_important_metrics
from utils.pickle_utils import load_from_pickle


X_eval, y_eval = get_eval_dataset()

baseline_model = load_from_pickle(
    BaselineModel.SAVED_BASELINE_MODEL_FILENAME, BaselineModel
)

prediction = baseline_model.predict(X_eval)

tn, fp, fn, tp = get_confusion_matrix_as_tuple(y_eval, prediction)

print("Evaluation results: ")
print_important_metrics(tp, tn, fp, fn)
