\# Customer Segmentation and Churn Prediction
**Live demo:** [Open the dashboard](https://....streamlit.app)



An end-to-end machine learning project that groups retail customers into segments (K-Means) and predicts which customers are likely to stop buying (Logistic Regression), with an interactive Streamlit dashboard.



\## Problem



A retailer wants to know:

1\. What types of customers do we have? (segmentation)

2\. Which customers are likely to leave, so we can send them a win-back offer? (churn prediction)



\## Dataset



\[Online Retail dataset (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/352/online+retail): transactions of a UK-based online store. Free to download. The raw file is not included in this repo because of its size.



\## Approach



1\. \*\*Data cleaning\*\* (`src/data\_cleaning.py`): removed rows with missing CustomerID, cancelled orders and invalid quantities or prices.

2\. \*\*EDA\*\* (`src/eda.py`): monthly revenue, top countries, customer spending, top products, orders by weekday.

3\. \*\*Feature engineering\*\* (`src/feature\_engineering.py`):

&#x20;  - The data is split at a cutoff date (last date minus 90 days).

&#x20;  - Features are built \*\*only from the past\*\* (before the cutoff): Recency, Frequency, Monetary, NumProducts, Tenure, AvgOrderValue.

&#x20;  - The churn label is built \*\*only from the future\*\* (after the cutoff): a customer who made no purchase in the last 90 days is marked `Churn = 1`.

&#x20;  - This split avoids data leakage.

4\. \*\*Segmentation\*\* (`src/segmentation.py`): K-Means on scaled Recency, Frequency and Monetary values. K was chosen using the elbow method and silhouette score.

5\. \*\*Churn model\*\* (`src/train\_final.py`): Logistic Regression in a scikit-learn Pipeline (log transform of skewed columns, StandardScaler, model). 80/20 stratified train-test split.

6\. \*\*Evaluation\*\* (`src/evaluate\_model.py`): accuracy, ROC-AUC, confusion matrix, classification report.

7\. \*\*Dashboard\*\* (`app/dashboard.py`): four tabs: Overview, Segments, Predict churn, At-risk customers.



\## Results



| Metric | Value |

|---|---|

| Test accuracy (threshold 0.5) | 0.677 |

| Test ROC-AUC | 0.726 |



The dashboard flags a customer as high risk at a probability threshold of 0.4, so that more real churners are caught (higher recall) at the cost of some false alarms.



| Confusion matrix | ROC curve |

|---|---|

| !\[Confusion matrix](reports/figures/08\_confusion\_matrix.png) | !\[ROC curve](reports/figures/09\_roc\_curve.png) |



!\[Customer segments](reports/figures/07\_segments.png)

\## Model comparison



Three models were compared using 5-fold stratified cross-validation on the training set, and then evaluated once on the held-out test set (same 80/20 split for all).



| Model | CV AUC (mean) | CV AUC (std) | Test AUC | Test accuracy (0.5) | Recall (0.4) | Precision (0.4) |

|---|---|---|---|---|---|---|

| Logistic Regression | 0.750 | 0.010 | 0.726 | 0.677 | 0.752 | 0.572 |

| Random Forest | 0.747 | 0.011 | 0.720 | 0.662 | 0.810 | 0.556 |

| Gradient Boosting | 0.743 | 0.011 | 0.711 | 0.638 | 0.776 | 0.560 |



The three models perform almost the same (the AUC differences are smaller than the cross-validation standard deviation). Logistic Regression was kept as the final model because it is simple, fast and easy to explain, and it does not lose to the more complex models.



The comparison script is `src/compare\_models.py`. Results are saved in `reports/model\_comparison.csv`.



\## Segments found



| Segment | Meaning |

|---|---|

| Champions | Recent, frequent, high-spending customers |

| Regulars (cooling off) | Used to buy regularly, but have slowed down |

| Recent low-value buyers | Bought recently but spent little |

| Lost one-time buyers | Bought once, long ago |



\## Dashboard



Run locally:



```bash

pip install -r requirements.txt

python -m streamlit run app/dashboard.py

```



Then open http://localhost:8501.



\## Project structure



```

customer-churn-project/

|-- app/dashboard.py          Streamlit dashboard

|-- src/                      data cleaning, EDA, features, models, evaluation

|-- models/                   saved model files (.pkl)

|-- data/processed/           customer features, segments, predictions

|-- reports/figures/          charts used in this README

|-- requirements.txt

```



\## How to reproduce



1\. Download the Online Retail dataset and place it in `data/raw/`.

2\. Run the scripts in `src/` in this order: `load\_data.py`, `data\_cleaning.py`, `eda.py`, `feature\_engineering.py`, `segmentation.py`, `train\_final.py`, `evaluate\_model.py`.

3\. Run the dashboard.

evaluate\_model.py compare\_models.py ... train\_final.py, evaluate\_model.py, compare\_models.py.







Run all scripts from the project root folder, because file paths are relative.



\## Limitations



\- ROC-AUC of 0.726 is a baseline. Random Forest and Gradient Boosting did not beat it, so better gains are more likely to come from new features (for example product categories or return rate) than from a more complex model.- Churn is defined by a fixed 90-day window, which is a business assumption.

\- The predictions CSV includes customers the model was trained on, so the "flagged" count is not a fair test score. Use the test metrics above for performance.

\- The dataset covers a single UK retailer, so results may not apply elsewhere.



\## Tech stack



Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Streamlit, joblib.

