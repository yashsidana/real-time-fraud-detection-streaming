import streamlit as st
import pandas as pd
import requests
import time
import os

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Real-Time Fraud Detection",
    layout="wide"
)

st.title("💳 Real-Time Fraud Detection Dashboard")
st.caption("Streaming live transactions and detecting fraud...")

# -----------------------------
# Resolve absolute dataset path
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")

# -----------------------------
# Load dataset (for simulation)
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# -----------------------------
# Mode selector
# -----------------------------
mode = st.radio(
    "Select Mode",
    ["📡 Live Streaming", "🧪 Test Unseen Transaction"]
)

# ============================================================
# MODE 1: LIVE STREAMING
# ============================================================
if mode == "📡 Live Streaming":

    st.subheader("📊 Live Transaction Stream")

    # Show sample of incoming data
    st.dataframe(df.drop(columns=["Class"], errors="ignore").head(10))

    st.markdown("### 🔍 Fraud Detection Results")

    # Stream a few transactions (not infinite loop)
    stream_sample = df.sample(10)

    for _, row in stream_sample.iterrows():
        payload = row.drop("Class", errors="ignore").to_dict()

        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=payload,
                timeout=5
            )
            prediction = response.json()["prediction"]

            if prediction == "Fraud":
                st.error("🚨 Fraudulent Transaction Detected")
            else:
                st.success("✅ Normal Transaction")

        except Exception as e:
            st.warning(f"Backend not responding: {e}")

        time.sleep(1)

# ============================================================
# MODE 2: MANUAL UNSEEN DATA TESTING
# ============================================================
elif mode == "🧪 Test Unseen Transaction":

    st.subheader("🧪 Manual Transaction Testing (Unseen Data)")
    st.info("Enter transaction feature values manually to test the trained model.")

    with st.form("manual_transaction_form"):

        time_val = st.number_input("Time", value=0.0)

        features = {}
        for i in range(1, 29):
            features[f"V{i}"] = st.number_input(
                f"V{i}",
                value=0.0,
                format="%.5f"
            )

        submit = st.form_submit_button("🔍 Predict Fraud")

        if submit:
            payload = {"Time": time_val}
            payload.update(features)

            try:
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json=payload,
                    timeout=5
                )

                result = response.json()["prediction"]

                if result == "Fraud":
                    st.error("🚨 Fraudulent Transaction Detected!")
                else:
                    st.success("✅ Normal Transaction")

            except Exception as e:
                st.error(f"Prediction failed: {e}")
