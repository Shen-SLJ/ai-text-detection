from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from utils.pickle_utils import save_to_pickle

TRAIN_DOCUMENTS = [
  "This is a sample text.", 
  "Here is another example of text. Text processing is fun!",
]
TRAIN_LABELS = [0, 1] 

SAVED_VECTORIZER_FILENAME = "baseline_vectorizer.pkl"
SAVED_MODEL_FILENAME = "baseline_model.pkl"

# Pre-processing: Vectorization 
vectorizer = CountVectorizer()

vectorizer.fit(TRAIN_DOCUMENTS)

save_to_pickle(vectorizer, SAVED_VECTORIZER_FILENAME)

X_train = vectorizer.transform(TRAIN_DOCUMENTS)

# SVM Model
linear_svm = LinearSVC()

linear_svm.fit(X_train, TRAIN_LABELS)

save_to_pickle(linear_svm, SAVED_MODEL_FILENAME)


# Reference paper: 
# Detection of AI-generated Text: An Experimental Study (2024)
