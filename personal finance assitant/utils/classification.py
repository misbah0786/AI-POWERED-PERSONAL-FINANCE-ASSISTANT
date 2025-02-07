import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained classification model
with open("models/expense_classifier.pkl", "rb") as file:
    classifier = pickle.load(file)

# Load the TF-IDF vectorizer
with open("models/tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

def predict_expense_category(description):
    """Predicts the category of an expense based on its description."""
    vectorized_text = vectorizer.transform([description])
    category = classifier.predict(vectorized_text)
    return category[0]
 
