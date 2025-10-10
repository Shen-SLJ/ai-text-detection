from typing import Iterable
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from candidate_model.stylometric.model import StylometricModel
from utils.pickle_utils import save_to_pickle
from utils.gpu_utils import is_gpu_available
from utils.vector_manip_utils import combine_spmatrix_with_nparrays
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from scipy.sparse import spmatrix


class EmbeddingStylometricModel:
    """A model combining stylometrics Linear SVM with sentence-embeddings with platt-scaled linear SVM to detect AI generated text.

    Sentence embedder: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    """

    SAVED_MODEL_FILENAME = "candidate_embedding_stylometric.pkl"

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

        self.embedder = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        ).to("cuda" if is_gpu_available() else "cpu")
        self.stylometric_model = StylometricModel(
            train_documents=train_documents,
            train_labels=train_labels,
            use_burstiness=use_burstiness,
            use_readibility_score=use_readibility_score,
            use_character_ngram=use_character_ngram,
            use_word_ngram=use_word_ngram,
        )
        self.classifier = CalibratedClassifierCV(
            estimator=LinearSVC(max_iter=10000), method="sigmoid"
        )

    def train(self) -> "EmbeddingStylometricModel":
        self.stylometric_model.train()

        features = self.__get_feature_representation(
            documents=self.train_documents, use_stylometric_training_values=True
        )
        self.classifier.fit(features, self.train_labels)

        return self

    def __get_feature_representation(
        self, documents: Iterable[str], use_stylometric_training_values: bool = False
    ) -> spmatrix:
        embeddings: ndarray = self.embedder.encode(
            sentences=documents,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        stylometric_features = self.stylometric_model.get_feature_representation(
            documents=documents, use_training_values=use_stylometric_training_values
        )

        combined_features = combine_spmatrix_with_nparrays(
            sparse_matrix=stylometric_features, nparrays=[embeddings]
        )

        return combined_features

    def save(self) -> "EmbeddingStylometricModel":
        save_to_pickle(self, self.SAVED_MODEL_FILENAME)

        return self

    def predict(self, documents: Iterable[str]) -> ndarray:
        features = self.__get_feature_representation(documents)

        return self.classifier.predict(features)

    def predict_probability(self, documents: Iterable[str]) -> ndarray:
        features = self.__get_feature_representation(documents)

        return self.classifier.predict_proba(features)
