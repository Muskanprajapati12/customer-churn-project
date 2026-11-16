import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SEGMENTS_CSV = Path("data/processed/customer_segments.csv")

df = pd.read_csv(SEGMENTS_CSV)

# 1. Choose features (CustomerID is just an ID, so it is NOT used)
FEATURES = ["Recency", "Frequency", "NumProducts", "Tenure", "AvgOrderValue"]
SKEWED = ["Frequency", "NumProducts", "AvgOrderValue"]

X = df[FEATURES].copy()
y = df["Churn"]

# 2. Log transform the skewed columns
X[SKEWED] = np.log1p(X[SKEWED])

# 3. Split FIRST (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train size:", X_train.shape, "| Test size:", X_test.shape)

# 4. Scale: learn from train only, then apply to both
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train the model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Quick accuracy check
train_acc = model.score(X_train_scaled, y_train)
test_acc = model.score(X_test_scaled, y_test)
baseline = max(y_test.mean(), 1 - y_test.mean())

print("\nBaseline (always guess majority class):", round(baseline, 3))
print("Train accuracy:", round(train_acc, 3))
print("Test accuracy :", round(test_acc, 3))

# 7. Which features matter?
coefs = pd.Series(model.coef_[0], index=FEATURES).sort_values()
print("\nCoefficients (negative = less churn, positive = more churn):")
print(coefs.round(3))