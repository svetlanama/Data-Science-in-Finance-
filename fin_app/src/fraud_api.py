from datetime import date, time, datetime
import pandas as pd

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
    # transaction_date, transaction_time,
    features = [
        'transaction_amount',
        'customer_age',
        'customer_balance',
        'transaction_date',
        'transaction_time',
    ]
    feature_importance = dict(zip(features, importance))
    return {"feature_importance": feature_importance}


class Transaction(BaseModel):
    transaction_amount: float
    customer_age: int
    customer_balance: float
    transaction_date: date
    transaction_time: time

@app.post("/predict/")
def predict_credit_score(data: Transaction):
    # Convert input data to numpy array
    print(">>>>> loaded_pipeline")
    print(loaded_pipeline)

    input_data = np.array([
        data.transaction_amount,
        data.customer_age,
        data.customer_balance,
\

    ]).reshape(1, -1)

    print(" ----- 1111 -----")
    # Make predictions
    prediction = loaded_pipeline.predict(input_data)
    print(" ----- 2222 -----")
    # Get probabilities for each class
    probabilities = loaded_pipeline.predict_proba(input_data)
    confidence = probabilities[0].tolist()
    print(" ----- 33333 -----")
    # SHAP (SHapley Additive exPlanations) values help explain the model’s prediction for each feature.
    path = f"{path_python_material}/data/2-intermediate/dsif11-X_train_scaled.npy"
    # X_train = pd.read_csv(path)

    # print(X_train.columns.tolist())

    X_train_scaled = np.load(path)
    print(" ----- 44444 -----")
    print(X_train_scaled.shape)
    # print(X_train_scaled.columns.tolist())
    explainer = shap.LinearExplainer(loaded_pipeline[1], X_train_scaled)
    shap_values = explainer.shap_values(input_data)
    print("SHAP", shap_values.tolist())

    return {
        "fraud_prediction": int(prediction[0]),
        "confidence": confidence,
        "shap_values": shap_values.tolist(),
        "features": [
            'transaction_amount',
            # 'transaction_hour',
            # 'transaction_dayofweek',
            'customer_age',
            'customer_balance'
        ]
    }


@app.post('/predict_automation')
def predict_automation(files_to_process:List[str]):

    # from conf.conf import landing_path_input_data, landing_path_output_data
    landing_path_input_data = "../data/4-stream/automation_in"
    landing_path_output_data = "../data/4-stream/automation_out"

    print(f"Files to process (beginning): {files_to_process}")
    # if '.DS_Store' in files_to_process:
    #     files_to_process.remove('.DS_Store')
    #     print(f"Files to process: {files_to_process}")

    input_data = pd.concat([pd.read_csv(landing_path_input_data + "/" + f) for f in files_to_process], ignore_index=True, sort=False)

    # # generate predictions
    input_data['pred_fraud'] = loaded_pipeline.predict(input_data)
    input_data['pred_proba_fraud'] = loaded_pipeline.predict_proba(input_data.drop(columns=['pred_fraud']))[:, 1]
    input_data['pred_proba_fraud'] = input_data['pred_proba_fraud'].apply(lambda x: round(x, 5))

    now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    input_data.to_csv(landing_path_output_data + "/api_tagged_" + now + ".csv", index=False)
    return {
        "Predictions saved in " + landing_path_output_data + "/api_tagged_" + now + ".csv"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8503) 