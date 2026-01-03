import streamlit as st
import pandas as pd
import requests
import time
import plotly.graph_objects as go

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(layout="wide")
st.title("📡 Real-Time Fraud Detection Dashboard")

# ==========================
# METRICS SECTION
# ==========================
st.header("📊 Model Evaluation Metrics")

metrics = requests.get(f"{BACKEND_URL}/metrics").json()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", round(metrics["accuracy"], 3))
col2.metric("Precision", round(metrics["precision"], 3))
col3.metric("Recall", round(metrics["recall"], 3))
col4.metric("F1 Score", round(metrics["f1"], 3))

# Confusion Matrix
st.subheader("Confusion Matrix")
st.dataframe(pd.DataFrame(
    metrics["confusion_matrix"],
    columns=["Pred Normal", "Pred Fraud"],
    index=["Actual Normal", "Actual Fraud"]
))

# ROC Curve
st.subheader("ROC Curve")
roc = metrics["roc_curve"]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=roc["fpr"],
    y=roc["tpr"],
    mode="lines",
    name=f"AUC = {roc['auc']:.3f}"
))
fig.add_trace(go.Scatter(
    x=[0,1], y=[0,1],
    mode="lines", line=dict(dash="dash"),
    name="Random"
))
fig.update_layout(
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate"
)
st.plotly_chart(fig, use_container_width=True)

# ==========================
# LIVE STREAM SECTION
# ==========================
st.header("📡 Live Transaction Stream")

df = pd.read_csv("data/creditcard.csv").sample(10)

for _, row in df.iterrows():
    payload = row.drop("Class").to_dict()

    res = requests.post(f"{BACKEND_URL}/predict", json=payload).json()
    prediction = res["prediction"]

    st.write(payload)

    if prediction == "Fraud":
        st.error("🚨 FRAUD DETECTED")
    else:
        st.success("✅ Normal Transaction")

    time.sleep(1)
