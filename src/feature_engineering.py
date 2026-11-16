import pandas as pd
from pathlib import Path

CLEAN_CSV = Path("data/processed/clean_retail.csv")
FEATURES_CSV = Path("data/processed/customer_features.csv")

df = pd.read_csv(CLEAN_CSV, parse_dates=["InvoiceDate"])

# 1. Cutoff date: 90 days before the last date in the data
last_date = df["InvoiceDate"].max()
cutoff = last_date - pd.Timedelta(days=90)
print("Last date in data:", last_date)
print("Cutoff date      :", cutoff)

# 2. Split into past and future
past = df[df["InvoiceDate"] < cutoff]
future = df[df["InvoiceDate"] >= cutoff]
print("Past rows:", len(past), "| Future rows:", len(future))

# 3. One row per customer, using ONLY past data
features = past.groupby("CustomerID").agg(
    LastPurchase=("InvoiceDate", "max"),
    FirstPurchase=("InvoiceDate", "min"),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum"),
    NumProducts=("StockCode", "nunique"),
).reset_index()

# 4. Extra columns
features["Recency"] = (cutoff - features["LastPurchase"]).dt.days
features["Tenure"] = (cutoff - features["FirstPurchase"]).dt.days
features["AvgOrderValue"] = features["Monetary"] / features["Frequency"]

# 5. Churn label, using ONLY future data
returned_customers = future["CustomerID"].unique()
features["Churn"] = (~features["CustomerID"].isin(returned_customers)).astype(int)

# 6. Drop date columns
features = features.drop(columns=["LastPurchase", "FirstPurchase"])

# 7. Save
features.to_csv(FEATURES_CSV, index=False)

print("\nShape:", features.shape)
print("\nFirst 5 rows:")
print(features.head())
print("\nChurn counts:")
print(features["Churn"].value_counts())
print("\nChurn rate:", round(features["Churn"].mean() * 100, 1), "%")
print("\nSummary:")
print(features.describe())