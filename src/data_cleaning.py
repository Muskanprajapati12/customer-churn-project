import pandas as pd
from pathlib import Path

RAW_CSV = Path("data/raw/online_retail.csv")
CLEAN_CSV = Path("data/processed/clean_retail.csv")

# 1. Load
df = pd.read_csv(RAW_CSV, dtype={"InvoiceNo": str, "StockCode": str},
                 parse_dates=["InvoiceDate"])
print("Start:", df.shape)

# 2. Remove exact duplicate rows
before = len(df)
df = df.drop_duplicates()
print("Duplicates removed:", before - len(df))

# 3. Remove rows with no CustomerID
before = len(df)
df = df.dropna(subset=["CustomerID"])
print("Rows without CustomerID removed:", before - len(df))

# 4. Remove cancelled invoices (InvoiceNo starts with 'C')
before = len(df)
df = df[~df["InvoiceNo"].str.startswith("C")]
print("Cancelled invoices removed:", before - len(df))

# 5. Remove rows with zero or negative Quantity / UnitPrice
before = len(df)
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
print("Bad quantity/price rows removed:", before - len(df))

# 6. Fix data types and create a new column
df["CustomerID"] = df["CustomerID"].astype(int)
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# 7. Save cleaned data
CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEAN_CSV, index=False)

print("\nFinal shape:", df.shape)
print("Unique customers:", df["CustomerID"].nunique())
print("Date range:", df["InvoiceDate"].min(), "to", df["InvoiceDate"].max())
print("\nMissing values left:")
print(df.isnull().sum())