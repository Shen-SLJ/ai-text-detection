from typing import Iterable
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from utils.pickle_utils import save_to_pickle
from numpy import ndarray


class BaselineModel:
    """A baseline model for text classification using CountVectorizer and LinearSVC.

    Model architecture based on - Detection of AI-generated Text: An Experimental Study [2024]
    https://ieeexplore.ieee.org/document/10731116
    """

    SAVED_MODEL_FILENAME = "baseline_model.pkl"

    def __init__(self, train_documents: Iterable[str], train_labels: Iterable[int]):
        self.train_documents = train_documents
        self.train_labels = train_labels

        self.vectorizer = CountVectorizer()
        self.linear_svm = LinearSVC(max_iter=10000)

    def train(self) -> "BaselineModel":
        self.vectorizer.fit(self.train_documents)

        X_train = self.vectorizer.transform(self.train_documents)

        self.linear_svm.fit(X_train, self.train_labels)

        return self

    def save(self) -> "BaselineModel":
        save_to_pickle(self, self.SAVED_MODEL_FILENAME)

        print(f"Baseline model saved to {self.SAVED_MODEL_FILENAME}")

        return self

    def predict(self, documents: Iterable[str]) -> ndarray:
        X = self.vectorizer.transform(documents)

        prediction = self.linear_svm.predict(X)

        return prediction
