from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

TRAIN_DOCUMENTS = [
  "This is a sample text.", 
  "Here is another example of text. Text processing is fun!",
]
TEST_DOCUMENTS = [
  "This is a sample text.",
  "Here is another example of text. Text processing is fun!"
]
TRAIN_LABELS = [0, 1] 

# Pre-processing: Vectorization 
vectorizer = CountVectorizer()

vectorizer.fit(TRAIN_DOCUMENTS)

X_train = vectorizer.transform(TRAIN_DOCUMENTS)


# SVM Model
linear_svm = LinearSVC()

linear_svm.fit(X_train, TRAIN_LABELS)

# Prediction
X_test = vectorizer.transform(TEST_DOCUMENTS)

prediction = linear_svm.predict(X_test)

print("Prediction:", prediction)


# Reference paper: 
# Detection of AI-generated Text: An Experimental Study (2024)
