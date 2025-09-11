import numpy as np
from typing import Iterable
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.svm import LinearSVC
from utils.pickle_utils import save_to_pickle
from utils.stylometric_utils import (
    get_sentence_burstiness_score,
    get_flesch_reading_ease_score,
)
from utils.matrix_manip_utils import combine_spmatrix, combine_spmatrix_with_1d_nparrays
from numpy import ndarray


class StylometricModel:
    """A model combining stylometrics with a Linear SVM to detect AI generated text.
    """

    SAVED_MODEL_FILENAME = "candidate_stylometric.pkl"

    def __init__(
        self,
        train_documents: Iterable[str],
        train_labels: Iterable[int],
        use_burstiness=False,
        use_readibility_score=False,
        use_character_ngram=False,
        use_word_ngram=False,
    ):
        self.train_documents = train_documents
        self.train_labels = train_labels

        self.use_burstiness = use_burstiness
        self.use_readibility_score = use_readibility_score
        self.use_character_ngram = use_character_ngram
        self.use_word_ngram = use_word_ngram

        self.ngram_vectorizer_word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 3), min_df=5, max_df=0.8
        )
        self.ngram_vectorizer_char = TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), min_df=5, max_df=0.8
        )
        self.classifier = LinearSVC(max_iter=10000)

    def train(self) -> "StylometricModel":
        if self.use_word_ngram:
            self.ngram_vectorizer_word.fit(self.train_documents)
        if self.use_character_ngram:
            self.ngram_vectorizer_char.fit(self.train_documents)

        X_train = self.__get_feature_representation(
            self.train_documents,
            use_burstiness=self.use_burstiness,
            use_readibility_score=self.use_readibility_score,
            use_character_ngram=self.use_character_ngram,
            use_word_ngram=self.use_word_ngram,
        )

        self.classifier.fit(X_train, self.train_labels)

        return self

    def __get_feature_representation(
        self,
        documents: Iterable[str],
        use_burstiness,
        use_readibility_score,
        use_character_ngram,
        use_word_ngram,
    ) -> ndarray:
        X = None

        if use_character_ngram or use_word_ngram:
            X_word = (
                self.ngram_vectorizer_word.transform(documents)
                if use_word_ngram
                else None
            )
            X_char = (
                self.ngram_vectorizer_char.transform(documents)
                if use_character_ngram
                else None
            )

            X = combine_spmatrix(X_word, X_char)

        if use_burstiness:
            burstiness_scores = [
                get_sentence_burstiness_score(document) for document in documents
            ]

            X = combine_spmatrix_with_1d_nparrays(X, [np.array(burstiness_scores)])

        if use_readibility_score:
            readibility_scores = [
                get_flesch_reading_ease_score(document) for document in documents
            ]

            X = combine_spmatrix_with_1d_nparrays(X, [np.array(readibility_scores)])

        if X is None:
            raise ValueError("At least one feature type must be selected.")

        return X

    def save(self) -> "StylometricModel":
        save_to_pickle(self, self.SAVED_MODEL_FILENAME)

        print(f"Stylometric with embedding model saved to {self.SAVED_MODEL_FILENAME}")

        return self

    def predict(self, documents: Iterable[str]) -> ndarray:
        X = self.__get_feature_representation(
            documents,
            use_burstiness=self.use_burstiness,
            use_readibility_score=self.use_readibility_score,
            use_character_ngram=self.use_character_ngram,
            use_word_ngram=self.use_word_ngram,
        )

        prediction = self.classifier.predict(X)

        return prediction
