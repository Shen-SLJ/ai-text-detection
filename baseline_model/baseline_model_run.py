from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from utils.pickle_utils import load_from_pickle

TEST_DOCUMENTS = [
  "This is a sample text.",
  "Here is another example of text. Text processing is fun!"
]

VECTORIZER_FILENAME_TO_LOAD = "baseline_vectorizer.pkl"
MODEL_FILENAME_TO_LOAD = "baseline_model.pkl"

# Prediction
vectorizer = load_from_pickle(VECTORIZER_FILENAME_TO_LOAD, CountVectorizer)
linear_svm = load_from_pickle(MODEL_FILENAME_TO_LOAD, LinearSVC)

X_test = vectorizer.transform(TEST_DOCUMENTS)

prediction = linear_svm.predict(X_test)

print("Prediction:", prediction)