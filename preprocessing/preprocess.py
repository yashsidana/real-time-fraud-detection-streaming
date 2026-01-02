import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/creditcard.csv")

print(df.head())
print(df.info())
print(df['Class'].value_counts())

df['Class'].value_counts().plot(kind='bar')
plt.title("Fraud vs Normal Transactions")
plt.show()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

df.dropna(inplace=True)
