from utils.pickle_utils import load_from_pickle
from baseline_model.baseline_model_train import BaselineModel

TEST_DOCUMENTS = [
  "This is a sample text.",
  "Here is another example of text. Text processing is fun!"
]

BASELINE_MODEL_FILENAME = "baseline_model.pkl"

# Prediction
baseline_model = load_from_pickle(BASELINE_MODEL_FILENAME, BaselineModel)
prediction = baseline_model.predict(TEST_DOCUMENTS)

print("Prediction:", prediction)