from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from utils.pickle_utils import load_from_pickle, save_to_pickle
import numpy as np

if __name__ == '__main__':
    X: np.ndarray = load_from_pickle("bert_embeddings.pkl")
    y: np.ndarray = load_from_pickle("labels.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    svm = LinearSVC()
    svm.fit(X=X_train, y=y_train)
    svm_score = svm.score(X=X_test, y=y_test)
    print(f"SVM Model accuracy: {svm_score:.2f}")

    save_to_pickle(svm, "svm_model.pkl")