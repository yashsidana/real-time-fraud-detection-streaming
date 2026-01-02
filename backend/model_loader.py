import joblib
import os

# Get absolute path to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model.pkl")

# Load trained model
model = joblib.load(MODEL_PATH)


def predict_fraud(transaction: dict):
    values = list(transaction.values())
    prediction = model.predict([values])[0]
    return "Fraud" if prediction == 1 else "Normal"
