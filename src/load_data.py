import pandas as pd
from pathlib import Path

# 1. Where the files live
RAW_XLSX = Path("data/raw/Online Retail.xlsx")
RAW_CSV = Path("data/raw/online_retail.csv")

# 2. Load the data
if RAW_CSV.exists():
    df = pd.read_csv(RAW_CSV, dtype={"InvoiceNo": str, "StockCode": str},
                     parse_dates=["InvoiceDate"])
    print("Loaded from CSV (fast).")
else:
    print("Reading Excel file. This can take 1-2 minutes, please wait...")
    df = pd.read_excel(RAW_XLSX, dtype={"InvoiceNo": str, "StockCode": str})
    df.to_csv(RAW_CSV, index=False)
    print("Saved a CSV copy so next time it loads quickly.")

# 3. First look
print("\n--- SHAPE (rows, columns) ---")
print(df.shape)

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- COLUMN TYPES AND NON-NULL COUNTS ---")
df.info()

print("\n--- MISSING VALUES PER COLUMN ---")
print(df.isnull().sum())

print("\n--- NUMBER SUMMARY ---")
print(df.describe())

print("\n--- DATE RANGE ---")
print(df["InvoiceDate"].min(), "to", df["InvoiceDate"].max())

print("\n--- UNIQUE CUSTOMERS ---")
print(df["CustomerID"].nunique())