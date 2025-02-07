import pandas as pd
import pickle

# Load pre-trained time-series forecasting model
with open("models/expense_forecaster.pkl", "rb") as file:
    model = pickle.load(file)

def forecast_expenses(data, months):
    """Forecasts future expenses for a specified number of months."""
    # Extract relevant time series data
    ts_data = data.set_index("Date")["Expense"]
    ts_data.index = pd.to_datetime(ts_data.index)
    
    # Generate forecast
    forecast = model.forecast(steps=months)
    return forecast