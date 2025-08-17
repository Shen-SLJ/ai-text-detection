from typing import List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from utils.pickle_utils import save_to_pickle

TRAIN_DOCUMENTS = [
    "This is a sample text.",
    "Here is another example of text. Text processing is fun!",
]
TRAIN_LABELS = [0, 1]

class BaselineModel:
    """A baseline model for text classification using CountVectorizer and LinearSVC.
    
    Model architecture based on - Detection of AI-generated Text: An Experimental Study [2024]
    https://ieeexplore.ieee.org/document/10731116
    """
    SAVED_BASELINE_MODEL_FILENAME = "baseline_model.pkl"

    def __init__(self, train_documents: list[str], train_labels: list[int]):
        self.train_documents = train_documents
        self.train_labels = train_labels

        self.vectorizer = CountVectorizer()
        self.linear_svm = LinearSVC()
      
    def train(self) -> "BaselineModel":
        self.vectorizer.fit(self.train_documents)
        
        X_train = self.vectorizer.transform(self.train_documents)

        self.linear_svm.fit(X_train, self.train_labels)

        return self

    def save(self):
        print(f"Baseline model saved to {self.SAVED_BASELINE_MODEL_FILENAME}")

        return save_to_pickle(self, self.SAVED_BASELINE_MODEL_FILENAME)

    def predict(self, documents: List[str]):
        X = self.vectorizer.transform(documents)

        prediction = self.linear_svm.predict(X)

        return prediction

BaselineModel(TRAIN_DOCUMENTS, TRAIN_LABELS).train().save()