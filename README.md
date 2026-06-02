# 📉 Customer Churn Prediction System

An end-to-end Machine Learning application that predicts whether a customer is likely to churn using customer demographics, account information, and service usage data.

The project includes:

- Data Analysis & Feature Engineering
- Machine Learning Model Training
- FastAPI Backend Deployment
- Streamlit Frontend Dashboard
- Real-time Churn Prediction

---

# 🚀 Project Overview

Customer churn is one of the most important business problems in subscription-based industries. Acquiring a new customer is often more expensive than retaining an existing one.

This project helps identify customers who are likely to leave the company, allowing businesses to take proactive retention measures.

---

# 🛠️ Tech Stack

### Machine Learning
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Uvicorn

### Deployment
- FastAPI
- Streamlit

### Visualization
- Matplotlib
- Seaborn

### Model Persistence
- Joblib

---

# 📊 Dataset

Dataset: Telco Customer Churn Dataset

Target Variable:

```text
Churn
```

Classes:

```text
0 → Customer Stays
1 → Customer Churns
```

---

# 🔍 Exploratory Data Analysis (EDA)

Performed analysis on:

- Customer tenure distribution
- Monthly charges distribution
- Total charges distribution
- Churn class distribution
- Contract type analysis
- Internet service analysis
- Payment method analysis

Key Findings:

- Month-to-month customers churn significantly more.
- Customers with Fiber Optic internet show higher churn rates.
- Longer tenure customers are less likely to churn.
- Electronic check users have relatively higher churn.

---

# ⚙️ Feature Engineering

Created additional business-oriented features:

### Number of Services

```python
num_services
```

Counts the total subscribed services.

### Monthly Contract Indicator

```python
is_monthly_contract
```

Identifies customers with month-to-month contracts.

### Average Monthly Charges

```python
avg_monthly_charges
```

Calculated using:

```python
TotalCharges / tenure
```

### Tenure Group

Categorized customers based on subscription duration.

---

# 🤖 Model Training

Models Evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

Final Selected Model:

```text
XGBoost Classifier
```

Reason:

- Best predictive performance
- Strong handling of non-linear relationships
- Excellent probability estimation

---

# 📈 Model Evaluation

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

Example Metrics:

```text
ROC-AUC Score: XX.XX
Accuracy: XX.XX
Precision: XX.XX
Recall: XX.XX
F1 Score: XX.XX
```

*(Replace with your actual values)*

---

# 🏗️ Project Architecture

```text
Streamlit Dashboard
          │
          ▼
      FastAPI
          │
          ▼
 Feature Engineering
          │
          ▼
 Preprocessor Pipeline
          │
          ▼
   XGBoost Model
          │
          ▼
 Prediction Result
```

---

# 📂 Project Structure

```text
customer-churn-prediction/

│
├── app.py                 # FastAPI Backend
├── streamlit_app.py       # Streamlit Frontend
│
├── model.pkl              # Trained Model
├── preprocessor.pkl       # Saved Preprocessor
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── requirements.txt
├── README.md
│
└── screenshots/
```

---

# 🚀 Running the Project

## Step 1: Clone Repository

```bash
git clone <repository-url>
cd customer-churn-prediction
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3: Start FastAPI Server

```bash
uvicorn app:app --reload
```

FastAPI Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Step 4: Launch Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Streamlit Dashboard:

```text
http://localhost:8501
```

---

# 📸 Application Screenshots

### Streamlit Dashboard

(Add screenshot here)

### Prediction Output

(Add screenshot here)

### FastAPI Swagger UI

(Add screenshot here)

---

# 🎯 Sample Prediction Output

```json
{
    "churn_probability": 0.8445,
    "churn": 1
}
```

Interpretation:

```text
Customer has an 84.45% probability of churning.
```

---

# 🔮 Future Improvements

- PostgreSQL Integration
- Docker Containerization
- Cloud Deployment
- Model Monitoring
- Automated Retraining Pipeline
- Customer Retention Recommendation System

---

# 👨‍💻 Author

Pratham Bisht

Aspiring Machine Learning Engineer focused on building production-ready ML systems using Python, FastAPI, SQL, and Cloud Technologies.
