import numpy as np
from typing import Iterable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from utils.pickle_utils import save_to_pickle
from utils.stylometric_utils import (
    get_sentence_burstiness_score,
    get_flesch_reading_ease_score,
)
from utils.vector_manip_utils import combine_spmatrix, combine_spmatrix_with_1d_nparrays
from utils.stylometric_utils import get_document_metrics_as_feature
from numpy import ndarray
from scipy.sparse import spmatrix


class CandidateStylometricModel:
    """A model combining stylometrics with a Linear SVM to detect AI generated text."""

    SAVED_MODEL_FILENAME = "candidate_stylometric.pkl"
    MAX_VOCAB_SIZE_WORD_NGRAM = 10000
    MAX_VOCAB_SIZE_CHAR_NGRAM = 75000

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
        self.burstiness_normalizer = StandardScaler()
        self.readibility_normalizer = StandardScaler()

        self.ngram_vectorizer_word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=5,
            max_df=0.9,
            max_features=self.MAX_VOCAB_SIZE_WORD_NGRAM,
        )
        self.ngram_vectorizer_char = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=5,
            max_df=0.9,
            max_features=self.MAX_VOCAB_SIZE_CHAR_NGRAM,
        )
        self.classifier = LinearSVC(max_iter=10000)

    def train(self) -> "CandidateStylometricModel":
        if self.use_word_ngram:
            self.__fit_word_n_gram_vectorizer()
        if self.use_character_ngram:
            self.__fit_char_n_gram_vectorizer()
        if self.use_burstiness:
            self.__fit_burstiness_normalizer()
        if self.use_readibility_score:
            self.__fit_readibility_normalizer()

        X_train = self.get_feature_representation(self.train_documents)

        self.classifier.fit(X_train, self.train_labels)

        return self

    def __fit_word_n_gram_vectorizer(self):
        self.ngram_vectorizer_word.fit(self.train_documents)

    def __fit_char_n_gram_vectorizer(self):
        self.ngram_vectorizer_char.fit(self.train_documents)

    def __fit_burstiness_normalizer(self):
        burstiness_scores = get_document_metrics_as_feature(
            self.train_documents, get_sentence_burstiness_score
        )
        self.burstiness_normalizer.fit(burstiness_scores)

    def __fit_readibility_normalizer(self):
        readibility_scores = get_document_metrics_as_feature(
            self.train_documents, get_flesch_reading_ease_score
        )
        self.readibility_normalizer.fit(readibility_scores)

    def get_feature_representation(self, documents: Iterable[str]) -> spmatrix:
        """Get stylometric features. Features are l2 normalised."""
        X = None

        if self.use_character_ngram or self.use_word_ngram:
            X_word = (
                self.ngram_vectorizer_word.transform(documents)
                if self.use_word_ngram
                else None
            )
            X_char = (
                self.ngram_vectorizer_char.transform(documents)
                if self.use_character_ngram
                else None
            )

            X = combine_spmatrix(X_word, X_char)

        if self.use_burstiness:
            burstiness_scores = self.burstiness_normalizer.transform(
                get_document_metrics_as_feature(
                    documents, get_sentence_burstiness_score
                )
            )

            X = combine_spmatrix_with_1d_nparrays(X, [burstiness_scores])

        if self.use_readibility_score:
            readibility_scores = self.readibility_normalizer.transform(
                get_document_metrics_as_feature(
                    documents, get_flesch_reading_ease_score
                )
            )

            X = combine_spmatrix_with_1d_nparrays(X, [readibility_scores])

        if X is None:
            raise ValueError("At least one feature type must be selected.")

        return X

    def save(self) -> "CandidateStylometricModel":
        save_to_pickle(self, self.SAVED_MODEL_FILENAME)

        print(f"Stylometric with embedding model saved to {self.SAVED_MODEL_FILENAME}")

        return self

    def predict(self, documents: Iterable[str]) -> ndarray:
        X = self.get_feature_representation(documents=documents)

        prediction = self.classifier.predict(X)

        return prediction
