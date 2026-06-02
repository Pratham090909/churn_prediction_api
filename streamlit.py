import streamlit as st
import requests

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Customer Churn Prediction Dashboard")
st.markdown(
    "Predict whether a customer is likely to churn based on their profile and services."
)

# -----------------------------
# INPUT SECTION
# -----------------------------

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    Partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    Dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12
    )

    PhoneService = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    MultipleLines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    InternetService = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    OnlineSecurity = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    OnlineBackup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col2:

    DeviceProtection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    TechSupport = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    StreamingTV = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    StreamingMovies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    Contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    PaperlessBilling = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    MonthlyCharges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    TotalCharges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0
    )

# -----------------------------
# PREDICT BUTTON
# -----------------------------

if st.button("🚀 Predict Churn"):

    payload = {
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
        "TotalCharges": TotalCharges
    }

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload
        )

        st.write(response.status_code)
        st.write(response.text)

        result = response.json()

        if "error" in result:
            st.error(result["error"])

        else:

            probability = result["churn_probability"]
            prediction = result["churn"]

            st.divider()

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    label="Churn Probability",
                    value=f"{probability:.2%}"
                )

            with c2:
                if prediction == 1:
                    st.metric(
                        label="Prediction",
                        value="Likely to Churn"
                    )
                else:
                    st.metric(
                        label="Prediction",
                        value="Likely to Stay"
                    )

            st.divider()

            if prediction == 1:
                st.error(
                    f"⚠️ High Risk Customer\n\n"
                    f"Predicted Churn Probability: {probability:.2%}"
                )
            else:
                st.success(
                    f"✅ Customer Likely to Stay\n\n"
                    f"Predicted Churn Probability: {probability:.2%}"
                )

    except Exception as e:
        st.error(f"Error: {str(e)}")
