from utils.dataset_preprocessor import DatasetPreprocessor
from utils.pickle_utils import save_to_pickle
from utils.bert_embedder import BERTEmbedder
import numpy as np

EMBEDDING_PICKLE_FILENAME = "bert_embeddings.pkl"
LABELS_PICKLE_FILENAME = "labels.pkl"

if __name__ == "__main__":
    data_preprocessor = DatasetPreprocessor()
    dataset_numpy = data_preprocessor.load_datasets().combined_dataset_as_numpy_array()
    features = dataset_numpy[:, 0]
    labels = dataset_numpy[:, 1]

    # Convert features and labels to lists
    embedder = BERTEmbedder()
    embedding = np.empty(shape=(len(features), 768))
    for i, feature in enumerate(features):
        embedding[i] = embedder.get_bert_embedding(feature)

    save_to_pickle(embedding, EMBEDDING_PICKLE_FILENAME)
    save_to_pickle(labels, LABELS_PICKLE_FILENAME)
