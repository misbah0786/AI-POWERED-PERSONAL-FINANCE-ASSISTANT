import streamlit as st
from utils.data_processing import load_data, preprocess_data
from utils.classification import predict_expense_category
from utils.regression import recommend_budget
from utils.forecasting import forecast_expenses

# App title
st.title("Personal Finance Assistant - AI Agent")

# Load and preprocess data
expense_data = load_data("data/expense_data.csv")
budget_data = load_data("data/budget_data.csv")
forecast_data = load_data("data/expense_forecast_data.csv")

# Sidebar for navigation
option = st.sidebar.selectbox(
    "Choose a feature:",
    ["Expense Categorization", "Budget Recommendations", "Expense Forecasting"]
)

if option == "Expense Categorization":
    st.header("Predictive Expense Categorization")
    description = st.text_input("Enter expense description:")
    if st.button("Classify Expense"):
        category = predict_expense_category(description)
        st.write(f"Predicted Category: **{category}**")

elif option == "Budget Recommendations":
    st.header("Budget Recommendations")
    income = st.number_input("Enter your monthly income:")
    if st.button("Get Budget Recommendation"):
        recommendation = recommend_budget(income, budget_data)
        st.write(f"Recommended Budget: **${recommendation:.2f}**")

elif option == "Expense Forecasting":
    st.header("Expense Forecasting")
    months = st.number_input("Enter months to forecast:", min_value=1, max_value=12)
    if st.button("Forecast Expenses"):
        forecast = forecast_expenses(forecast_data, months)
        st.line_chart(forecast)
 
