import pickle
import numpy as np

# Load the trained regression model
with open("models/budget_regressor.pkl", "rb") as file:
    regressor = pickle.load(file)

def recommend_budget(income, data):
    """Recommends a budget based on income and past data."""
    # Use the regression model to predict budget allocation
    avg_savings = data['Savings'].mean()
    recommendation = regressor.predict([[income, avg_savings]])
    return recommendation[0]
 
