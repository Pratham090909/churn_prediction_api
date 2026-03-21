# Customer Churn Prediction API

An end-to-end Machine Learning system that predicts customer churn using XGBoost and is deployed as a REST API using FastAPI.

---

## Live Demo

- 🔗 **API Base URL:**  
  https://churn-api-upw2.onrender.com  

- 📄 **Interactive API Docs (Swagger UI):**  
  https://churn-api-upw2.onrender.com/docs  

👉 Use the `/docs` link to test predictions directly in your browser.

---

## Features

- Data preprocessing & feature engineering
- Model training using XGBoost
- Handling class imbalance using SMOTE
- Hyperparameter tuning (RandomizedSearchCV)
- Model explainability using SHAP
- Deployment using FastAPI on cloud

---

## Model Overview

The model predicts the probability of customer churn based on:

- Demographics (gender, senior citizen, etc.)
- Account information (tenure, contract type)
- Services used (internet, streaming, security, etc.)
- Billing details (monthly & total charges)

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- SHAP
- FastAPI
- Uvicorn

---

## Project Structure

