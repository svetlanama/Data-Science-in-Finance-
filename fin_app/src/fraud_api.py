import os
from datetime import date, time, datetime
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import shap
import pickle
from typing import Dict, List, Optional, Union
import json
from enum import Enum

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
        data.customer_balance, \
 \
        ]).reshape(1, -1)

    # Make predictions
    prediction = loaded_pipeline.predict(input_data)

    # Get probabilities for each class
    probabilities = loaded_pipeline.predict_proba(input_data)
    confidence = probabilities[0].tolist()

    # SHAP (SHapley Additive exPlanations) values help explain the model's prediction for each feature.
    path = f"{path_python_material}/data/2-intermediate/dsif11-X_train_scaled.npy"

    X_train_scaled = np.load(path)
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
def predict_automation(files_to_process: List[str]):
    # from conf.conf import landing_path_input_data, landing_path_output_data
    landing_path_input_data = "../data/4-stream/automation_in"
    landing_path_output_data = "../data/4-stream/automation_out"

    print(f"Files to process (beginning): {files_to_process}")
    # if '.DS_Store' in files_to_process:
    #     files_to_process.remove('.DS_Store')
    #     print(f"Files to process: {files_to_process}")

    input_data = pd.concat([pd.read_csv(landing_path_input_data + "/" + f) for f in files_to_process],
                           ignore_index=True, sort=False)

    # # generate predictions
    input_data['pred_fraud'] = loaded_pipeline.predict(input_data)
    input_data['pred_proba_fraud'] = loaded_pipeline.predict_proba(input_data.drop(columns=['pred_fraud']))[:, 1]
    input_data['pred_proba_fraud'] = input_data['pred_proba_fraud'].apply(lambda x: round(x, 5))

    now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    input_data.to_csv(landing_path_output_data + "/api_tagged_" + now + ".csv", index=False)
    return {
        "Predictions saved in " + landing_path_output_data + "/api_tagged_" + now + ".csv"
    }


class TransactionDayAnalysis(BaseModel):
    transaction_date: date
    transaction_time: time
    transaction_amount: float
    customer_id: str


class ProcessingType(str, Enum):
    DAILY_SUMMARY = "daily_summary"
    CUSTOMER_PATTERNS = "customer_patterns"
    ANOMALY_DETECTION = "anomaly_detection"


# @app.post("/analyze-transaction-day")
# def analyze_transaction_day(data: TransactionDayAnalysis):
#     """
#     Analyzes transaction patterns based on day and time
#     """
#     # Convert date to day of week (0 = Monday, 6 = Sunday)
#     day_of_week = data.transaction_date.weekday()
#     hour = data.transaction_time.hour
#
#     # Basic analysis
#     is_weekend = day_of_week >= 5
#     is_business_hours = 9 <= hour <= 17
#
#     # Risk assessment based on time patterns
#     risk_score = 0
#     if is_weekend:
#         risk_score += 0.2
#     if not is_business_hours:
#         risk_score += 0.3
#
#     return {
#         "day_of_week": day_of_week,
#         "hour": hour,
#         "is_weekend": is_weekend,
#         "is_business_hours": is_business_hours,
#         "time_based_risk_score": risk_score,
#         "recommendation": "High risk" if risk_score > 0.4 else "Normal risk"
#     }


@app.post("/process")
def process_transactions():
    print(" ----- 1111 -----")
    files_to_process = ["to_process.csv"]  # ["to_score_2024_06_13_12_15_32.csv"]  #
    processing_type = ProcessingType.ANOMALY_DETECTION
    start_date = "2024-01-01"
    end_date = "2024-04-06"

    landing_path_input_data = "../data/4-stream/automation_in"
    landing_path_output_data = "../data/4-stream/automation_out"

    # Initialize a list to hold individual DataFrames
    dfs = []

    for file in files_to_process:
        full_path = f"{landing_path_input_data}/{file}"
        if os.path.exists(full_path):
            df = pd.read_csv(full_path)
            dfs.append(df)
        else:
            print(f"File not found: {full_path}")

    # Concatenate all data into a single DataFrame
    if dfs:
        input_data = pd.concat(dfs, ignore_index=True, sort=False)
        print("✅ Input data loaded successfully")
    else:
        input_data = pd.DataFrame()
        print("⚠️ No files loaded; input_data is empty")

    if 'transaction_date' in input_data.columns:
        input_data['transaction_date'] = pd.to_datetime(input_data['transaction_date'])

    result = None
    daily_stats = input_data.groupby('transaction_date')['transaction_amount'].agg(['mean', 'std']).reset_index()
    input_data = input_data.merge(daily_stats, on='transaction_date')
    input_data['z_score'] = (input_data['transaction_amount'] - input_data['mean']) / input_data['std']
    result = input_data[input_data['z_score'].abs() > 4.0]
    anomalies = result

    return {
        "message": "Processing completed",
        "anomalies_found": len(anomalies),
        "anomalies": anomalies.to_dict(orient="records")
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8503)
