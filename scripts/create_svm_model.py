from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from utils.pickle_utils import load_from_pickle, save_to_pickle
import numpy as np

if __name__ == '__main__':
    X: np.ndarray = load_from_pickle("bert_embeddings.pkl")
    y: np.ndarray = load_from_pickle("labels.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    svm = LinearSVC()
    svm.fit(X=X_train, y=y_train)

    # save the model
    save_to_pickle(svm, "svm_model.pkl")

    # evaluate the model
    y_pred = svm.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_true=y_test, y_pred=y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true=y_test, y_pred=y_pred))
