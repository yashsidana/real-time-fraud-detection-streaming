from fastapi import FastAPI
from backend.model_loader import predict_fraud, get_metrics

app = FastAPI()

@app.post("/predict")
def predict(transaction: dict):
    result = predict_fraud(transaction)
    return {"prediction": result}

@app.get("/metrics")
def metrics():
    return get_metrics()
