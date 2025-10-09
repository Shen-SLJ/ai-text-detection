from candidate_model.stylometric.model import CandidateStylometricModel
from dataset_processing.datasets import get_eval_dataset
from utils.metric_utils import print_important_metrics, get_confusion_matrix_as_tuple
from utils.pickle_utils import load_from_pickle


X_eval, y_eval = get_eval_dataset()

candidate_model = load_from_pickle(
    CandidateStylometricModel.SAVED_MODEL_FILENAME, CandidateStylometricModel
)

prediction = candidate_model.predict(X_eval)

tn, fp, fn, tp = get_confusion_matrix_as_tuple(y_eval, prediction)

print("Evaluation results: ")
print_important_metrics(tp, tn, fp, fn)
