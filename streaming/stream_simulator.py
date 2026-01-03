import pandas as pd
import time
import json

df = pd.read_csv("../data/creditcard.csv")

for _, row in df.iterrows():
    transaction = row.to_dict()
    print(json.dumps(transaction))
    time.sleep(0.3)  # simulates real-time delay
