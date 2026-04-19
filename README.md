# 🛡️ Fraud Risk Prioritization System

End-to-end Machine Learning system for **fraud prediction, investigation prioritization, drift monitoring, and interpretable risk scoring**.

This project simulates how financial institutions use ML to **rank suspicious transactions**, allowing investigators to focus on the most important cases instead of manually reviewing everything.

---

## 🚀 Key Highlights

✔ Time-aware validation split (realistic deployment simulation)
✔ Advanced feature engineering for fraud signals
✔ LightGBM model optimized for imbalanced tabular data
✔ Ranking-based evaluation (Precision@K)
✔ Drift monitoring for production reliability
✔ SHAP explainability for model transparency

✔ FastAPI inference API

✔ Streamlit fraud analyst dashboard


---

## 🎯 Problem Statement

Fraud investigation teams cannot review every transaction manually.

This system helps answer:

* Which transactions are most suspicious?
* Which transactions should be reviewed first?
* How effective is the prioritization strategy?
* Why does the model consider a transaction risky?

---

## 📊 Model Performance

| Metric        | Value      |
| ------------- | ---------- |
| ROC-AUC       | **0.8138** |
| PR-AUC        | **0.2026** |
| Precision@100 | **0.61**   |

Fraud rate ≈ **3.5%**

Precision@100 = **61%**, meaning **61 of top 100 flagged transactions are actual fraud**, ~17× better than random selection.

---

## 🧠 Feature Engineering

Key engineered signals:

* log-transformed transaction amount
* frequency encoding for high-cardinality variables
* card-level behavioral aggregates
* time-based transaction patterns
* distance anomaly features

These help the model capture **behavioral deviations**, a strong indicator of fraud.

---

## 🖥️ Dashboard

### Fraud Analyst Console

![Dashboard](outputs\plots\dashboard.png)

---

### Prediction Example

![Prediction](outputs\plots\prediction_example.png)

---

### SHAP Explainability

![SHAP](outputs\plots\shap_summary.png)

---

## 🧱 Architecture

```
Raw Transaction Data
↓
Feature Engineering Pipeline
↓
LightGBM Fraud Model
↓
Fraud Probability Score
↓
Ranking Layer (Precision@K)
↓
Threshold Optimization
↓
Drift Monitoring
↓
FastAPI Inference API
↓
Streamlit Dashboard
```

---

## ⚙️ Tech Stack

* Python
* scikit-learn
* LightGBM
* SHAP
* FastAPI
* Streamlit
* pandas
* numpy

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
```

Place dataset in:

```
data/raw/
```

Required files:

* train_transaction.csv
* train_identity.csv

Run:

```bash
python -m src.data.make_splits
python -m src.models.train_lgbm
python -m src.ranking.evaluate_ranking
uvicorn api.main:app --reload
streamlit run dashboard/app.py
```

---

## 💡 Why This Project Stands Out

* decision-focused evaluation metrics
* business-aware threshold tuning
* interpretable ML predictions
* drift-aware monitoring
* deployable ML system design

---

## 🔭 Future Improvements

* batch inference pipeline
* automated retraining
* cloud deployment

---

## 👤 Author

Flagship ML Engineer portfolio project focused on:

* feature engineering depth
* model reliability
* real-world decision support systems
