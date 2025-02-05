import pandas as pd
import numpy as np
import openai
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv
import os

# Load dataset
file_path = "/mnt/data/Budget.csv"
df = pd.read_csv(file_path)

# Convert categories to numerical values
df["Category"] = df["Category"].astype("category").cat.codes  # Encode text categories to numbers

# Features and target variable
X = df[["Category"]]
y = df["Budget"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Function to predict budget for a given category
def predict_budget(category_name):
    category_code = df[df["Category"] == category_name].index[0]  # Convert category to numerical code
    predicted_budget = model.predict(np.array([[category_code]]))[0]
    return round(predicted_budget, 2)

# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Chatbot function
def chat_with_ai(user_message):
    prompt = f"You are a finance assistant. Help users with budgeting.\n\nUser: {user_message}\nAI:"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": user_message}]
    )
    
    return response["choices"][0]["message"]["content"]

# Example usage
category = "Coffee Shops"
print(f"Predicted budget for {category}: ${predict_budget(category)}")

user_query = "How much should I allocate for entertainment?"
print(chat_with_ai(user_query))
