from utils.pickle_utils import load_from_pickle

if __name__ == '__main__':
    load_from_pickle("bert_embeddings.pkl")
    load_from_pickle("labels.pkl")
    
# svm = LinearSVC()
# svm.fit(X=get_bert_embedding(text), y=labels)

# test_text = ["good", "bad"]
# print(svm.predict(get_bert_embedding(test_text)))

# Helpful path management
# https://stackoverflow.com/a/68299898