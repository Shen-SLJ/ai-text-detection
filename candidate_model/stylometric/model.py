import numpy as np
from xgboost import XGBClassifier
from typing import Iterable
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC
from utils.pickle_utils import save_to_pickle
from utils.stylometric_utils import (
    get_sentence_burstiness_score,
    get_flesch_reading_ease_score,
)
from utils.matrix_manip_utils import combine_spmatrix_with_1d_nparrays
from numpy import ndarray


class StylometricModel:
    """A model combining stylometrics with an SVM to detect AI generated text.

    Args:
        classifier: The classifier to use. Should be an sklearn compatible classifier.
    """

    SAVED_MODEL_FILENAME = "candidate_stylometric.pkl"

    def __init__(self, train_documents: Iterable[str], train_labels: Iterable[int]):
        self.train_documents = train_documents
        self.train_labels = train_labels

        self.ngram_vectorizer = HashingVectorizer(ngram_range=(3, 5))
        self.classifier = LinearSVC(max_iter=10000)
        # self.classifier = XGBClassifier(objective="binary:hinge", device="cuda")

    def train(self) -> "StylometricModel":
        self.ngram_vectorizer.fit(self.train_documents)

        X_train = self.__get_feature_representation(self.train_documents)

        self.classifier.fit(X_train, self.train_labels)

        return self

    def __get_feature_representation(self, documents: Iterable[str]) -> ndarray:
        X = self.ngram_vectorizer.transform(documents)

        burstiness_scores = [
            get_sentence_burstiness_score(document) for document in documents
        ]
        readibility_scores = [
            get_flesch_reading_ease_score(document) for document in documents
        ]

        X = combine_spmatrix_with_1d_nparrays(
            X, [np.array(burstiness_scores), np.array(readibility_scores)]
        )

        return X

    def save(self) -> "StylometricModel":
        save_to_pickle(self, self.SAVED_MODEL_FILENAME)

        print(f"Stylometric with embedding model saved to {self.SAVED_MODEL_FILENAME}")

        return self

    def predict(self, documents: Iterable[str]) -> ndarray:
        X = self.__get_feature_representation(documents)

        prediction = self.classifier.predict(X)

        return prediction
