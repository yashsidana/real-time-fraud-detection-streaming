import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# Always resolve paths relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model.pkl")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Split features and target
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, MODEL_PATH)

print("✅ Model trained and saved at:", MODEL_PATH)
