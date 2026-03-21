from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

model = pickle.load(open("churn_model.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

def feature_engineering(df):
    df = df.copy()

    df['num_services'] = (
        df[['PhoneService','MultipleLines','OnlineSecurity',
            'OnlineBackup','DeviceProtection','TechSupport',
            'StreamingTV','StreamingMovies']] == 'Yes'
    ).sum(axis=1)

    df['is_monthly_contract'] = (df['Contract'] == 'Month-to-month').astype(int)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    df['tenure'] = df['tenure'].replace(0, 1)

    df['avg_monthly_charges'] = df['TotalCharges'] / df['tenure']

    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 60, 72],
        labels=['0-1 yr', '1-2 yr', '2-4 yr', '4-5 yr', '5-6 yr']
    )

    return df


# Home route
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running 🚀"}


# Prediction route
@app.post("/predict")
def predict(data: dict):
    try:
        
        df = pd.DataFrame([data])
        df = feature_engineering(df)
        df = df.reindex(columns=preprocessor.feature_names_in_, fill_value=0)
        processed = preprocessor.transform(df)
        prob = model.predict_proba(processed)[0][1]

        return {
            "churn_probability": float(prob),
            "churn": int(prob > 0.4)
        }

    except Exception as e:
        return {"error": str(e)}