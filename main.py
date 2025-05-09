from sklearn.svm import LinearSVC
from utils.pickle_utils import load_from_pickle
from utils.bert_embedder import BERTEmbedder

text = [
    ""
]

if __name__ == '__main__':
    embedder = BERTEmbedder()

    svm: LinearSVC = load_from_pickle("svm_model.pkl")
    
    input = embedder.get_embedding(text)
    print(svm.predict(input))


# Helpful path management
# https://stackoverflow.com/a/68299898