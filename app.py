# ==========================================================
# STREAMLIT APP: LOAN APPROVAL PREDICTION
# Task 2 Random Forest Model
# Compatible with:
# Pandas 2.2.3, Numpy 1.26.4, Sklearn 1.3.0, Streamlit 1.52.0
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Loan Approval Prediction App")
st.write("Enter applicant details to predict loan approval status using a trained Random Forest model.")

# -------------------- LOAD MODEL & ENCODERS --------------------
@st.cache_data
def load_model():
    with open("random_forest_loan_approval_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("loan_label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

rf_model, label_encoders = load_model()

# -------------------- USER INPUTS --------------------
st.header("Applicant Information")

no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0)
education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed?", options=["Yes", "No"])
income_annum = st.number_input("Annual Income (in ₹)", min_value=0, step=100000, value=5000000)
loan_amount = st.number_input("Loan Amount (in ₹)", min_value=0, step=100000, value=1000000)
loan_term = st.number_input("Loan Term (in Years)", min_value=1, max_value=30, value=10)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)
residential_assets_value = st.number_input("Residential Assets Value (in ₹)", min_value=0, step=100000, value=1000000)
commercial_assets_value = st.number_input("Commercial Assets Value (in ₹)", min_value=0, step=100000, value=1000000)
luxury_assets_value = st.number_input("Luxury Assets Value (in ₹)", min_value=0, step=100000, value=500000)
bank_asset_value = st.number_input("Bank Asset Value (in ₹)", min_value=0, step=100000, value=2000000)

# -------------------- PREPARE INPUT FOR MODEL --------------------
input_data = pd.DataFrame({
    "no_of_dependents": [no_of_dependents],
    "education": [education],
    "self_employed": [self_employed],
    "income_annum": [income_annum],
    "loan_amount": [loan_amount],
    "loan_term": [loan_term],
    "cibil_score": [cibil_score],
    "residential_assets_value": [residential_assets_value],
    "commercial_assets_value": [commercial_assets_value],
    "luxury_assets_value": [luxury_assets_value],
    "bank_asset_value": [bank_asset_value]
})

# Encode categorical columns
for col in ["education", "self_employed"]:
    le = label_encoders[col]
    input_data[col] = le.transform(input_data[col])

# -------------------- PREDICTION --------------------
if st.button("Predict Loan Approval"):
    prediction = rf_model.predict(input_data)[0]
    prediction_proba = rf_model.predict_proba(input_data)[0]

    st.subheader("Prediction Result:")
    st.success(f"The loan is likely to be **{prediction}**.")

    st.subheader("Prediction Probabilities:")
    for cls, prob in zip(rf_model.classes_, prediction_proba):
        st.write(f"{cls}: {prob*100:.2f}%")
