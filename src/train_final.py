import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

SEGMENTS_CSV = Path("data/processed/customer_segments.csv")
PRED_CSV = Path("data/processed/customer_predictions.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.4

df = pd.read_csv(SEGMENTS_CSV)

FEATURES = ["Recency", "Frequency", "NumProducts", "Tenure", "AvgOrderValue"]
SKEWED = ["Frequency", "NumProducts", "AvgOrderValue"]

X = df[FEATURES]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

log_step = ColumnTransformer(
    [("log", FunctionTransformer(np.log1p), SKEWED)],
    remainder="passthrough",
)

pipeline = Pipeline([
    ("log", log_step),
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])

pipeline.fit(X_train, y_train)

y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Test ROC-AUC :", round(roc_auc_score(y_test, y_prob), 3))

joblib.dump(pipeline, MODEL_DIR / "churn_pipeline.pkl")
joblib.dump({"features": FEATURES, "threshold": THRESHOLD},
            MODEL_DIR / "churn_config.pkl")

df["ChurnProb"] = pipeline.predict_proba(df[FEATURES])[:, 1]
df["PredictedChurn"] = (df["ChurnProb"] >= THRESHOLD).astype(int)
df.to_csv(PRED_CSV, index=False)

print("\nSaved: churn_pipeline.pkl, churn_config.pkl, customer_predictions.csv")
print("Customers flagged as likely to churn:", df["PredictedChurn"].sum())