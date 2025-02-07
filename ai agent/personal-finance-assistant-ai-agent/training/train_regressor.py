import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Load dataset
data = pd.read_csv("data/budget_data.csv")

# Features and target
X = data[['Income', 'Savings']]
y = data['Expense']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Evaluate the model
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the model
with open("models/budget_regressor.pkl", "wb") as file:
    pickle.dump(regressor, file)

print("Budget regressor saved!")