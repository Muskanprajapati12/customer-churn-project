import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

CLEAN_CSV = Path("data/processed/clean_retail.csv")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CLEAN_CSV, parse_dates=["InvoiceDate"])
sns.set_theme(style="whitegrid")

# ---------- Basic facts ----------
print("Rows:", len(df))
print("Customers:", df["CustomerID"].nunique())
print("Invoices:", df["InvoiceNo"].nunique())
print("Total revenue:", round(df["TotalPrice"].sum(), 2))

# ---------- 1. Monthly revenue ----------
df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
monthly = df.groupby("Month")["TotalPrice"].sum()

plt.figure(figsize=(10, 5))
monthly.plot(marker="o")
plt.title("Monthly Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIG_DIR / "01_monthly_revenue.png")
plt.close()

# ---------- 2. Top 10 countries by revenue ----------
country_rev = df.groupby("Country")["TotalPrice"].sum().nlargest(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=country_rev.values, y=country_rev.index)
plt.title("Top 10 Countries by Revenue")
plt.xlabel("Revenue")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_top_countries.png")
plt.close()

# ---------- 3. Customer spending distribution ----------
customer_spend = df.groupby("CustomerID")["TotalPrice"].sum()
print("\nCustomer spending summary:")
print(customer_spend.describe())

plt.figure(figsize=(10, 5))
sns.histplot(np.log1p(customer_spend), bins=40)
plt.title("Customer Total Spending (log scale)")
plt.xlabel("log(1 + total spend)")
plt.ylabel("Number of customers")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_customer_spending.png")
plt.close()

# ---------- 4. Top 10 products by quantity ----------
top_products = df.groupby("Description")["Quantity"].sum().nlargest(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_products.values, y=top_products.index)
plt.title("Top 10 Products by Quantity Sold")
plt.xlabel("Quantity sold")
plt.ylabel("")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_top_products.png")
plt.close()

# ---------- 5. Orders by weekday ----------
df["Weekday"] = df["InvoiceDate"].dt.day_name()
order_days = ["Monday", "Tuesday", "Wednesday", "Thursday",
              "Friday", "Saturday", "Sunday"]
orders_per_day = (df.groupby("Weekday")["InvoiceNo"].nunique()
                    .reindex(order_days).fillna(0))

plt.figure(figsize=(10, 5))
sns.barplot(x=orders_per_day.index, y=orders_per_day.values)
plt.title("Number of Orders by Weekday")
plt.xlabel("")
plt.ylabel("Unique invoices")
plt.tight_layout()
plt.savefig(FIG_DIR / "05_orders_by_weekday.png")
plt.close()

print("\nAll charts saved in", FIG_DIR)