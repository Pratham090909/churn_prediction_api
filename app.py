import streamlit as st
import pandas as pd
import pickle

st.title("Churn Prediction App 🚀")

# Load model + preprocessor
model = pickle.load(open("churn_model.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

st.sidebar.header("Customer Details")

# Inputs
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

tenure = st.sidebar.slider("Tenure", 0, 72)

PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])

StreamingTV = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

MonthlyCharges = st.sidebar.number_input("Monthly Charges")
TotalCharges = st.sidebar.number_input("Total Charges")

# Derived features (VERY IMPORTANT 🔥)
num_services = 0
services = [PhoneService, MultipleLines, InternetService, OnlineSecurity,
            OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies]

for s in services:
    if s == "Yes":
        num_services += 1

is_monthly_contract = 1 if Contract == "Month-to-month" else 0
avg_monthly_charges = MonthlyCharges / tenure if tenure > 0 else 0

# tenure_group (example logic)
if tenure < 12:
    tenure_group = "0-1 year"
elif tenure < 24:
    tenure_group = "1-2 years"
else:
    tenure_group = "2+ years"

# Prediction
if st.button("Predict"):

    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "num_services": num_services,
        "is_monthly_contract": is_monthly_contract,
        "avg_monthly_charges": avg_monthly_charges,
        "tenure_group": tenure_group
    }])

    input_processed = preprocessor.transform(input_data)
    prediction = model.predict(input_processed)[0]
    prob = model.predict_proba(input_processed)[0][1]

    if prediction == 1:
        st.error(f"Customer will churn ❌ (Probability: {prob:.2f})")
    else:
        st.success(f"Customer will stay ✅ (Probability: {prob:.2f})")

