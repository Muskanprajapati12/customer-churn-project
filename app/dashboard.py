from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ---------- Paths (work no matter where you run the app from) ----------
BASE = Path(__file__).resolve().parent.parent
DATA_CSV = BASE / "data" / "processed" / "customer_predictions.csv"
MODEL_PATH = BASE / "models" / "churn_pipeline.pkl"
CONFIG_PATH = BASE / "models" / "churn_config.pkl"

# Names for the 4 segments (based on the profile table from segmentation.py)
SEGMENT_NAMES = {
    1: "Champions",
    2: "Regulars (cooling off)",
    3: "Recent low-value buyers",
    0: "Lost one-time buyers",
}

st.set_page_config(page_title="Customer Churn Dashboard",
                   page_icon="📊", layout="wide")


# ---------- Load data and model once (cached) ----------
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_CSV)
    data["SegmentName"] = data["Segment"].map(SEGMENT_NAMES)
    return data


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), joblib.load(CONFIG_PATH)


df = load_data()
pipeline, config = load_model()
FEATURES = config["features"]
THRESHOLD = config["threshold"]

# ---------- Header and KPI cards ----------
st.title("📊 Customer Segmentation and Churn Dashboard")
st.caption("Online Retail dataset | K-Means segments | Logistic Regression churn model")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", f"{len(df):,}")
k2.metric("Actual churn rate", f"{df['Churn'].mean() * 100:.1f}%")
k3.metric("Flagged as high-risk", f"{df['PredictedChurn'].sum():,}")
k4.metric("Revenue (past window)", f"£{df['Monetary'].sum():,.0f}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Segments", "Predict churn", "At-risk customers"]
)

# ---------- Tab 1: Overview ----------
with tab1:
    st.subheader("Churned vs stayed customers")
    label = df["Churn"].map({0: "Stayed", 1: "Churned"})
    left, right = st.columns(2)
    with left:
        st.write("Number of customers")
        st.bar_chart(label.value_counts())
    with right:
        st.write("Average behaviour of each group")
        summary = (df.groupby(label)[["Recency", "Frequency", "Monetary", "NumProducts"]]
                     .mean().round(1))
        st.dataframe(summary)
    st.caption("Churned customers usually order less often and have fewer different products.")

# ---------- Tab 2: Segments ----------
with tab2:
    st.subheader("Customer segments (K-Means on Recency, Frequency, Monetary)")
    profile = df.groupby("SegmentName").agg(
        Customers=("CustomerID", "count"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgMonetary=("Monetary", "mean"),
        ChurnRate=("Churn", "mean"),
    ).round(2)
    profile["ChurnRate"] = (profile["ChurnRate"] * 100).round(1)
    st.dataframe(profile)

    c1, c2 = st.columns(2)
    with c1:
        st.write("Customers per segment")
        st.bar_chart(profile["Customers"])
    with c2:
        st.write("Churn rate (%) per segment")
        st.bar_chart(profile["ChurnRate"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, group in df.groupby("SegmentName"):
        ax.scatter(group["Recency"], group["Monetary"], s=12, alpha=0.6, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("Recency (days since last purchase)")
    ax.set_ylabel("Monetary (total spend, log scale)")
    ax.set_title("Segments: Recency vs Monetary")
    ax.legend()
    st.pyplot(fig)

# ---------- Tab 3: Predict churn for one customer ----------
with tab3:
    st.subheader("Check the churn risk of one customer")
    left, right = st.columns(2)
    with left:
        recency = st.slider("Days since last purchase (Recency)", 0, 283, 60)
        frequency = st.slider("Number of orders (Frequency)", 1, 50, 3)
        tenure = st.slider("Days since first purchase (Tenure)", 0, 283, 150)
    with right:
        num_products = st.slider("Different products bought", 1, 200, 30)
        avg_order = st.number_input("Average order value (£)", min_value=1.0,
                                    max_value=5000.0, value=300.0, step=10.0)

    if tenure < recency:
        st.warning("Tenure should not be smaller than Recency "
                   "(first purchase cannot be after the last purchase).")

    if st.button("Predict churn risk"):
        new_customer = pd.DataFrame([{
            "Recency": recency,
            "Frequency": frequency,
            "NumProducts": num_products,
            "Tenure": tenure,
            "AvgOrderValue": avg_order,
        }])[FEATURES]
        prob = float(pipeline.predict_proba(new_customer)[0, 1])
        st.metric("Churn probability", f"{prob * 100:.1f}%")
        st.progress(min(max(prob, 0.0), 1.0))
        if prob >= THRESHOLD:
            st.error(f"High risk (above the {THRESHOLD:.0%} threshold). "
                     "Consider a win-back offer.")
        else:
            st.success("Lower risk. This customer is likely to stay.")

# ---------- Tab 4: At-risk customers ----------
with tab4:
    st.subheader("High-value customers likely to churn")
    risky = df[df["PredictedChurn"] == 1].sort_values("Monetary", ascending=False)
    show_cols = ["CustomerID", "SegmentName", "Recency", "Frequency",
                 "Monetary", "ChurnProb"]
    table = risky[show_cols].copy()
    table["ChurnProb"] = table["ChurnProb"].round(3)

    top_n = st.slider("How many customers to show", 10, 100, 20)
    st.dataframe(table.head(top_n))
    st.download_button("Download full at-risk list (CSV)",
                       table.to_csv(index=False),
                       file_name="at_risk_customers.csv",
                       mime="text/csv")
