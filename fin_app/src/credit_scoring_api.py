from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import shap
import pickle
from typing import Dict, List, Optional
import json

app = FastAPI()

# Load the pipeline
path_python_material = ".."
model_id = "lr1"
with open(f"{path_python_material}/models/{model_id}-pipeline.pkl", "rb") as f:
    loaded_pipeline = pickle.load(f)

# Define the input data model
class CreditData(BaseModel):
    age: int
    income: float
    employment_length: int
    credit_history_length: int
    debt_to_income: float
    credit_utilization: float
    number_of_credit_lines: int
    number_of_delinquencies: int

# Load the model and feature names
# try:
#     model = joblib.load('../models/model.pkl')
#     scaler = joblib.load('../models/scaler.pkl')
#     feature_names = [
#         'age', 'income', 'employment_length', 'credit_history_length',
#         'debt_to_income', 'credit_utilization', 'number_of_credit_lines',
#         'number_of_delinquencies'
#     ]
#     print("Successfully loaded existing model and scaler")
# except FileNotFoundError:
#     print("Model file not found. Using dummy model for demonstration.")
#     # Create a dummy model for demonstration
#     class DummyModel:
#         def predict(self, X):
#             return np.random.randint(300, 850, size=(X.shape[0],))
#         def predict_proba(self, X):
#             return np.random.rand(X.shape[0], 1)
#     model = DummyModel()
#     scaler = None
#     feature_names = [
#         'age', 'income', 'employment_length', 'credit_history_length',
#         'debt_to_income', 'credit_utilization', 'number_of_credit_lines',
#         'number_of_delinquencies'
#     ]

# Initialize SHAP explainer
# explainer = shap.TreeExplainer(model)

@app.get("/")
def read_root():
    return {"message": "Credit Scoring API is running"}

@app.get("/feature-importance")
def get_feature_importance():
    importance = loaded_pipeline[1].coef_[0].tolist()
    print("importance")
    print(importance)
    features = ['transaction_a  mount', 'customer_age', 'customer_balance']
    feature_importance = dict(zip(features, importance))
    print("feature_importance")
    print(feature_importance)
    return {"feature_importance": feature_importance}
    # For demonstration, return some dummy feature importance values
    # In a real application, you would calculate this from your model
    # feature_importance = {
    #     'debt_to_income': 0.25,
    #     'credit_utilization': 0.20,
    #     'number_of_delinquencies': 0.15,
    #     'credit_history_length': 0.12,
    #     'employment_length': 0.10,
    #     'income': 0.08,
    #     'age': 0.06,
    #     'number_of_credit_lines': 0.04
    # }
    # return {"feature_importance": feature_importance}

@app.post("/predict/")
def predict_credit_score(data: CreditData):
    # Convert input data to numpy array
    input_data = np.array([
        data.age,
        data.income,
        data.employment_length,
        data.credit_history_length,
        data.debt_to_income,
        data.credit_utilization,
        data.number_of_credit_lines,
        data.number_of_delinquencies
    ]).reshape(1, -1)
    
    # Scale the input data if scaler is available
    if scaler is not None:
        input_data = scaler.transform(input_data)
    
    # Get prediction
    credit_score = int(model.predict(input_data)[0])
    
    # Get confidence (for demonstration, using a random value)
    confidence = np.random.rand()
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(input_data)
    
    return {
        "credit_score": credit_score,
        "confidence": confidence,
        "shap_values": shap_values[0].tolist(),
        "features": feature_names
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8503) 