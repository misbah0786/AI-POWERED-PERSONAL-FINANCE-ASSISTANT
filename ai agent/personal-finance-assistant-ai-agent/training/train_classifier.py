import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# Load your dataset
data = pd.read_csv("data/expense_data.csv")

# Preprocess data
X = data['Description']
y = data['Category']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numerical data using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the classifier
classifier = LogisticRegression()
classifier.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
with open("models/expense_classifier.pkl", "wb") as file:
    pickle.dump(classifier, file)

with open("models/tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("Expense classifier and vectorizer saved!")