import json
import joblib
import sys

model = joblib.load("../model/fraud_model.pkl")

for line in sys.stdin:
    transaction = json.loads(line.strip())
    features = list(transaction.values())[:-1]
    prediction = model.predict([features])

    if prediction[0] == 1:
        print("🚨 FRAUD DETECTED:", transaction["Amount"])
