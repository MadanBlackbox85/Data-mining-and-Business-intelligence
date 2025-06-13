from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
text = [
    "This is good",      # Positive
    "This is awesome",   # Positive
    "I hate this",       # Negative
    "This is terrible",  # Negative
    "This is great",     # Positive
    "This is worst"      # Negative
]
labels = ['positive', 'positive', 'negative', 'negative', 'positive', 'negative']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text)
X_train = X[:4]  
y_train = labels[:4]
X_test = X[4:]  
y_test = labels[4:]
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("True Labels:",y_test)
print("Predicted Labels:",list(y_pred))
