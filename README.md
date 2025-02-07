Finance Assistant – AI-Powered Budget Prediction

📖 Description:
Finance Assistant is a machine learning-based tool that predicts expenses based on an individual's income and savings. It helps users better manage their finances by providing AI-driven insights into their spending habits. The project uses Linear Regression to model expense trends and make data-driven predictions.

Features:
✔ Predicts expenses based on income & savings
✔ Uses scikit-learn for machine learning
✔ Saves trained model as a pickle (.pkl) file for reuse
✔ Provides Mean Squared Error (MSE) for model evaluation
✔ Easy-to-use and can be expanded for more financial insights

🛠️ Installation & Setup:
1️⃣ Clone the repository:

bash
Copy
Edit

cd finance-assistant
2️⃣ Install dependencies:

bash
Copy
Edit
pip install pandas scikit-learn pickle-mixin
3️⃣ Run the script:

bash
Copy
Edit
python finance_assistant.py
📂 Project Structure:
bash
Copy
Edit
/finance_assistant
│── data/
│   └── budget_data.csv  # Dataset file
│── models/
│   └── finance_model.pkl  # Saved trained model
│── finance_assistant.py  # Main ML script
│── README.md  # Project documentation

📊 How It Works:
Loads financial dataset (budget_data.csv)
Splits data into training and testing sets
Trains a Linear Regression model
Evaluates model accuracy using Mean Squared Error (MSE)
Saves trained model (finance_model.pkl) for future use
