import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FEATURES_CSV = Path("data/processed/customer_features.csv")
SEGMENTS_CSV = Path("data/processed/customer_segments.csv")
FIG_DIR = Path("reports/figures")
MODEL_DIR = Path("models")
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

K_FINAL = 4   # change after looking at the elbow/silhouette chart

df = pd.read_csv(FEATURES_CSV)

# 1. RFM columns, log transform (reduces skew), then scale
rfm = df[["Recency", "Frequency", "Monetary"]]
rfm_log = np.log1p(rfm)
scaler = StandardScaler()
X = scaler.fit_transform(rfm_log)

# 2. Try K = 2..8
k_values = list(range(2, 9))
inertias = []
silhouettes = []
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels))
    print(f"K={k}  inertia={km.inertia_:.1f}  silhouette={silhouettes[-1]:.3f}")

# 3. Elbow and silhouette charts
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(k_values, inertias, marker="o")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of clusters (K)")
axes[0].set_ylabel("Inertia")
axes[1].plot(k_values, silhouettes, marker="o", color="green")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of clusters (K)")
axes[1].set_ylabel("Score")
plt.tight_layout()
plt.savefig(FIG_DIR / "06_choose_k.png")
plt.close()

# 4. Final model
final_km = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
df["Segment"] = final_km.fit_predict(X)

# 5. Profile each segment
profile = df.groupby("Segment").agg(
    Customers=("CustomerID", "count"),
    AvgRecency=("Recency", "mean"),
    AvgFrequency=("Frequency", "mean"),
    AvgMonetary=("Monetary", "mean"),
    ChurnRate=("Churn", "mean"),
).round(2)
profile["ChurnRate"] = (profile["ChurnRate"] * 100).round(1)
print("\nSegment profile:")
print(profile)

# 6. Scatter plot of segments
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x="Recency", y="Monetary",
                hue="Segment", palette="deep", alpha=0.7)
plt.yscale("log")
plt.title("Customer Segments (Recency vs Monetary)")
plt.tight_layout()
plt.savefig(FIG_DIR / "07_segments.png")
plt.close()

# 7. Save results and fitted objects
df.to_csv(SEGMENTS_CSV, index=False)
joblib.dump(scaler, MODEL_DIR / "rfm_scaler.pkl")
joblib.dump(final_km, MODEL_DIR / "kmeans_model.pkl")
print("\nSaved segments, scaler and model.")