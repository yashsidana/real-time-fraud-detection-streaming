import joblib
import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")

# Load model
model = joblib.load(MODEL_PATH)

# Load dataset
df = pd.read_csv(DATA_PATH)
X = df.drop("Class", axis=1)
y = df["Class"]

# Predictions
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# Metrics
fpr, tpr, _ = roc_curve(y, y_prob)

metrics = {
    "accuracy": accuracy_score(y, y_pred),
    "precision": precision_score(y, y_pred),
    "recall": recall_score(y, y_pred),
    "f1": f1_score(y, y_pred),
    "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    "roc_curve": {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": auc(fpr, tpr)
    }
}

def predict_fraud(transaction: dict):
    ordered_values = [transaction[col] for col in X.columns]
    pred = model.predict([ordered_values])[0]
    return "Fraud" if pred == 1 else "Normal"

def get_metrics():
    return metrics
