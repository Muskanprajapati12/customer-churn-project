import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve,
                             precision_score, recall_score)

SEGMENTS_CSV = Path("data/processed/customer_segments.csv")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SEGMENTS_CSV)

# Same setup as churn_model.py (Monetary removed)
FEATURES = ["Recency", "Frequency", "NumProducts", "Tenure", "AvgOrderValue"]
SKEWED = ["Frequency", "NumProducts", "AvgOrderValue"]

X = df[FEATURES].copy()
y = df["Churn"]
X[SKEWED] = np.log1p(X[SKEWED])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions: class (0/1) and probability of churn
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# 1. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print("Confusion matrix:")
print(cm)
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

# 2. Precision, recall, F1
print("\nClassification report:")
print(classification_report(y_test, y_pred,
                            target_names=["Stayed (0)", "Churned (1)"]))

# 3. ROC-AUC
auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC:", round(auc, 3))

# 4. What happens if we change the threshold?
print("\nThreshold  Precision  Recall")
for t in [0.3, 0.4, 0.5, 0.6]:
    pred_t = (y_prob >= t).astype(int)
    p = precision_score(y_test, pred_t)
    r = recall_score(y_test, pred_t)
    print(f"{t:<10} {p:<10.3f} {r:.3f}")

# 5. Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred Stay", "Pred Churn"],
            yticklabels=["Actual Stay", "Actual Churn"])
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(FIG_DIR / "08_confusion_matrix.png")
plt.close()

# 6. Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"Model (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "09_roc_curve.png")
plt.close()

print("\nCharts saved in", FIG_DIR)