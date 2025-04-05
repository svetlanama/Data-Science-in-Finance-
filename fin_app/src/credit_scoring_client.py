import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# API endpoint
api_url = "http://127.0.0.1:8502"

st.title("Credit Scoring App")

# Display site header
image_path = "../images/dsif header 2.jpeg"
try:
    img = Image.open(image_path)
    st.image(img, use_column_width=True)
except FileNotFoundError:
    st.error(f"Image not found at {image_path}. Please check the file path.")

# Input fields for credit scoring
st.header("Customer Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100)
    income = st.number_input("Annual Income ($)", min_value=0)
    employment_length = st.number_input("Employment Length (years)", min_value=0)
    credit_history_length = st.number_input("Credit History Length (years)", min_value=0)

with col2:
    debt_to_income = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, max_value=100.0)
    credit_utilization = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0)
    number_of_credit_lines = st.number_input("Number of Credit Lines", min_value=0)
    number_of_delinquencies = st.number_input("Number of Delinquencies (past 2 years)", min_value=0)

# Prepare data for prediction
data = {
    "age": age,
    "income": income,
    "employment_length": employment_length,
    "credit_history_length": credit_history_length,
    "debt_to_income": debt_to_income,
    "credit_utilization": credit_utilization,
    "number_of_credit_lines": number_of_credit_lines,
    "number_of_delinquencies": number_of_delinquencies
}

if st.button("Show Credit Risk Factors (Feature Importance)"):
    response = requests.get(f"{api_url}/feature-importance")
    print("Show Credit Risk Factors  REsponse")
    print(response.json())
    feature_importance = response.json().get('feature_importance', {})

    features = list(feature_importance.keys())
    importance = list(feature_importance.values())

    fig, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_xlabel('Importance')
    ax.set_title('Credit Risk Factors')
    st.pyplot(fig)

if st.button("Get Credit Score"):
    response = requests.post(f"{api_url}/predict/", json=data)
    result = response.json()
    score = result['credit_score']
    confidence = result['confidence']

    # Display credit score
    st.subheader(f"Credit Score: {score}")
    
    # Credit score range visualization
    ranges = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    score_ranges = [300, 580, 670, 740, 800, 850]
    score_index = next(i for i, x in enumerate(score_ranges) if x > score) - 1
    
    fig, ax = plt.subplots()
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    ax.bar(ranges, [1]*5, color=colors)
    ax.set_title('Credit Score Range')
    ax.set_xticks(range(len(ranges)))
    ax.set_xticklabels(ranges)
    ax.axvline(x=score_index, color='black', linestyle='--')
    st.pyplot(fig)

if st.button("Show Credit Score Factors"):
    response = requests.post(f"{api_url}/predict/", json=data)
    result = response.json()
    
    # Extract SHAP values and feature names
    shap_values = np.array(result['shap_values'])
    features = result['features']

    # Display SHAP values
    st.subheader("Credit Score Factors Explanation")

    # Bar plot for SHAP values
    fig, ax = plt.subplots()
    ax.barh(features, shap_values[0])
    ax.set_xlabel('Impact on Credit Score')
    st.pyplot(fig) 