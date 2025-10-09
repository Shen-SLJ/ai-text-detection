from typing import Iterable
from sklearn.svm import LinearSVC
from candidate_model.stylometric.model import CandidateStylometricModel
from utils.pickle_utils import save_to_pickle
from utils.gpu_utils import is_gpu_available
from utils.matrix_manip_utils import combine_spmatrix_with_1d_nparrays
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from scipy.sparse import spmatrix


class EmbeddingStylometricModel:
    """A model combining stylometrics SVM with sentence-embeddings with SVM to detect AI generated text.

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
        self.stylometric_model = CandidateStylometricModel(
            train_documents=train_documents,
            train_labels=train_labels,
            use_burstiness=use_burstiness,
            use_readibility_score=use_readibility_score,
            use_character_ngram=use_character_ngram,
            use_word_ngram=use_word_ngram,
        )
        self.classifier = LinearSVC(max_iter=10000)

    def train(self) -> "EmbeddingStylometricModel":
        self.stylometric_model.train()

        features = self.__get_feature_representation(self.train_documents)

        self.classifier.fit(features, self.train_labels)

    def __get_feature_representation(self, documents: Iterable[str]) -> spmatrix:
        self.embeddings: ndarray = self.embedder.encode(
            sentences=documents,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        self.stylometric_features = self.stylometric_model.get_feature_representation(
            documents=documents
        )

        combined_features = combine_spmatrix_with_1d_nparrays(
            sparse_matrix=self.stylometric_features, nparrays=[self.embeddings]
        )

        return combined_features

    def save(self) -> "CandidateStylometricModel":
        save_to_pickle(self, self.SAVED_MODEL_FILENAME)

        return self

    def predict(self, documents: Iterable[str]) -> ndarray:
        features = self.__get_feature_representation(documents)

        return self.classifier.predict(features)


EmbeddingStylometricModel(train_documents=[], train_labels=[])
