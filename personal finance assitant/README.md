
# Personal Finance Assistant

An AI-powered tool for managing personal finances with features like budget recommendations, expense categorization, and forecasting. Developed to enhance my AI/ML skills.

----------

## Features

-   **Predictive Expense Categorization**: Classifies expenses based on transaction descriptions.
-   **Budget Recommendations**: Suggests monthly budgets based on income and past expenses.
-   **Expense Forecasting**: Predicts future expenses using time-series analysis.

----------

## Tech Stack

-   **Backend**: Python
-   **ML Libraries**: NumPy, Pandas, Scikit-learn, Statsmodels
-   **Interface**: Streamlit

----------

## Steps to Run

1. (Optional) Add more data in the `data` folder
2. (Optional)Train models:
    
    ```bash
    python training/train_classifier.py
    python training/train_regressor.py
    python training/train_forecaster.py 
    ```
3.  Run the app:
    
    ```bash
    streamlit run main.py
    ```
    
4.  Access the app at `http://localhost:8501`.

----------

## Machine Learning Models

-   **Predictive Expense Categorization**: Logistic Regression
-   **Budget Recommendations**: Linear Regression
-   **Expense Forecasting**: Exponential Smoothing (Time-Series)

----------

## License

[MIT](LICENSE) License. 
