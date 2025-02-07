import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle

# Load your dataset
data = pd.read_csv("data/expense_forecast_data.csv")

# Preprocess data
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

# Ensure data is sorted by date
data = data.sort_index()

# Extract the Expense column for time-series forecasting
ts_data = data['Expense']

# Train the forecasting model
model = ExponentialSmoothing(ts_data, seasonal='add', seasonal_periods=12).fit()

# Save the model
with open("models/expense_forecaster.pkl", "wb") as file:
    pickle.dump(model, file)

print("Expense forecasting model saved!")