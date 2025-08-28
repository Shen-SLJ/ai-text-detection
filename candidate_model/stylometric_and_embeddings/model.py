from typing import Iterable
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC
from utils.pickle_utils import save_to_pickle
from numpy import ndarray


class StylometricWithEmbeddingModel:
    """A model combining stylometrics, and RoBERTa embeddings together with an SVM to detect AI generated text.
    """

    SAVED_MODEL_FILENAME = "stylometric_and_embeddings.pkl"

    def __init__(self, train_documents: Iterable[str], train_labels: Iterable[int]):
        self.train_documents = train_documents
        self.train_labels = train_labels

        self.ngram_vectorizer = HashingVectorizer(ngram_range=(3, 5))
        self.linear_svm = LinearSVC(max_iter=10000)


    def train(self) -> "StylometricWithEmbeddingModel":
        self.ngram_vectorizer.fit(self.train_documents)

        X_train = self.ngram_vectorizer.transform(self.train_documents)

        self.linear_svm.fit(X_train, self.train_labels)

        return self

    def save(self) -> "StylometricWithEmbeddingModel":
        save_to_pickle(self, self.SAVED_MODEL_FILENAME)

        print(f"Stylometric with embedding model saved to {self.SAVED_MODEL_FILENAME}")

        return self

    def predict(self, documents: Iterable[str]) -> ndarray:
        X = self.ngram_vectorizer.transform(documents)

        prediction = self.linear_svm.predict(X)

        return prediction
