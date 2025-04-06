import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# API endpoint
api_url = "http://127.0.0.1:8502"

st.title("Credit Scoring App")

# Display site header
image_path = "../images/ai-protection.png"
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
    transaction_amount = st.number_input("Transaction Amount ($)", min_value=0)
    customer_balance = st.number_input("Customer Balance ($)", min_value=0)

with col2:
    debt_to_income = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, max_value=100.0)

data = {
    "customer_age": age,
    "transaction_amount": transaction_amount,
    "customer_balance": customer_balance,
}

if st.button("Display Feature Importance"):
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

if st.button("Predict Fraud "):
    response = requests.post(f"{api_url}/predict/", json=data)
    result = response.json()

    if result['fraud_prediction'] == 0:
        st.write("Prediction: Not Fraud")
    else:
        st.write("Prediction: Fraud")

    confidence = result['confidence']

    # Display  score
    # st.subheader(f" Score: {score}")
    
    # Credit score range visualization
    # ranges = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    # score_ranges = [300, 580, 670, 740, 800, 850]
    # score_index = next(i for i, x in enumerate(score_ranges) if x > score) - 1
    #
    labels = ['Not Fraudulent', 'Fraudulent']
    fig, ax = plt.subplots()
    ax.bar(labels, confidence, color=['green', 'red'])
    ax.set_ylabel('Confidence')
    ax.set_title('Prediction Confidence')
    st.pyplot(fig)
    # # colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    # # ax.bar(ranges, [1]*5, color=colors)
    # # ax.set_title('Score Range')
    # # ax.set_xticks(range(len(ranges)))
    # # ax.set_xticklabels(ranges)
    # # ax.axvline(x=score_index, color='black', linestyle='--')
    # st.pyplot(fig)

if st.button("Show Fraud Score Factors (Bar Plot)"):
    response = requests.post(f"{api_url}/predict/", json=data)
    result = response.json()
    
    # Extract SHAP values and feature names
    shap_values = np.array(result['shap_values'])
    features = result['features']

    # Display SHAP values
    st.subheader("Fraud Score Factors Explanation")

    # Bar plot for SHAP values
    fig, ax = plt.subplots()
    ax.barh(features, shap_values[0])
    ax.set_xlabel('Impact on Credit Score')
    st.pyplot(fig)

if st.button("Show Fraud Score Factors (Pie Plot)"):
    response = requests.post(f"{api_url}/predict/", json=data)
    result = response.json()

    # Extract SHAP values and feature names
    # shap_values = np.array(result['shap_values'])
    shap_values = np.array(result['shap_values'])[0]
    features = result['features']

    st.subheader("SHAP Pie Chart")
    # Take absolute values for pie chart
    shap_abs = np.abs(shap_values)

    fig, ax = plt.subplots()
    ax.pie(
        shap_abs,
        labels=features,
        autopct='%1.1f%%',
        startangle=90
    )
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)