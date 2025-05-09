from sklearn.svm import LinearSVC
from utils.pickle_utils import load_from_pickle
from utils.bert_embedder import BERTEmbedder

text = [
    "Cars are one of the most important inventions of the current century, and that is mainly because they are so useful for us humans.",
    "Is this really that good? Because I don't think so.",
    "I love this product! It works great and is very easy to use.",
    "Cars should be banned from cities because they cause pollution and traffic jams.",
]

if __name__ == '__main__':
    embedder = BERTEmbedder()

    svm: LinearSVC = load_from_pickle("svm_model.pkl")
    
    input = embedder.get_embedding(text)
    print(svm.predict(input))


# Helpful path management
# https://stackoverflow.com/a/68299898