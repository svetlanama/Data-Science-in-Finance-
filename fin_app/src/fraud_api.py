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


@app.get("/")
def read_root():
    return {"message": "Credit Scoring API is running"}

@app.get("/feature-importance")
def get_feature_importance():
    importance = loaded_pipeline[1].coef_[0].tolist()

    features = ['transaction_a  mount', 'customer_age', 'customer_balance']
    feature_importance = dict(zip(features, importance))
    return {"feature_importance": feature_importance}


class Transaction(BaseModel):
    transaction_amount: float
    customer_age: int
    customer_balance: float

@app.post("/predict/")
def predict_credit_score(data: Transaction):
    # Convert input data to numpy array

    input_data = np.array([
        data.transaction_amount,
        data.customer_age,
        data.customer_balance
    ]).reshape(1, -1)


    # Make predictions
    prediction = loaded_pipeline.predict(input_data)

    # Get probabilities for each class
    probabilities = loaded_pipeline.predict_proba(input_data)
    confidence = probabilities[0].tolist()

    # Shap values
    # SHAP (SHapley Additive exPlanations) values help explain the model’s prediction for each feature.
    path = f"{path_python_material}/data/2-intermediate/dsif11-X_train_scaled.npy"
    print(path)
    X_train_scaled = np.load(path)
    explainer = shap.LinearExplainer(loaded_pipeline[1], X_train_scaled)
    shap_values = explainer.shap_values(input_data)
    print("SHAP", shap_values.tolist())

    return {
        "fraud_prediction": int(prediction[0]),
        "confidence": confidence,
        "shap_values": shap_values.tolist(),
        "features": ['transaction_amount', 'customer_age', 'customer_balance']
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8503) 