import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score

SEGMENTS_CSV = Path("data/processed/customer_segments.csv")
OUT_CSV = Path("reports/model_comparison.csv")

FEATURES = ["Recency", "Frequency", "NumProducts", "Tenure", "AvgOrderValue"]
SKEWED = ["Frequency", "NumProducts", "AvgOrderValue"]

df = pd.read_csv(SEGMENTS_CSV)
X = df[FEATURES]
y = df["Churn"]

# Same split as train_final.py, so results are comparable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


def logistic_pipeline():
    log_step = ColumnTransformer(
        [("log", FunctionTransformer(np.log1p), SKEWED)],
        remainder="passthrough",
    )
    return Pipeline([
        ("log", log_step),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])


# Tree models do not need scaling or log transform
models = {
    "Logistic Regression": logistic_pipeline(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=20,
        random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []

for name, model in models.items():
    # 5-fold cross-validation on the TRAIN part only
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_mean = scores["test_score"].mean()
    cv_std = scores["test_score"].std()

    # Fit on full train, evaluate once on the untouched test set
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred_05 = (prob >= 0.5).astype(int)
    pred_04 = (prob >= 0.4).astype(int)

    rows.append({
        "Model": name,
        "CV_AUC_mean": round(cv_mean, 3),
        "CV_AUC_std": round(cv_std, 3),
        "Test_AUC": round(roc_auc_score(y_test, prob), 3),
        "Test_Accuracy(0.5)": round(accuracy_score(y_test, pred_05), 3),
        "Test_Recall(0.4)": round(recall_score(y_test, pred_04), 3),
        "Test_Precision(0.4)": round(precision_score(y_test, pred_04), 3),
    })

results = pd.DataFrame(rows)
print(results.to_string(index=False))

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUT_CSV, index=False)
print("\nSaved:", OUT_CSV)