from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import lightgbm as lgb
from fastapi import FastAPI

from api.schemas import TransactionInput
from src.features.build_features import transform_features
from src.utils.io_utils import load_json


app = FastAPI(title="Fraud Risk Prioritization API")


MODEL_PATH = "outputs/models/lgbm_model.txt"
ARTIFACTS_PATH = "outputs/models/feature_artifacts.json"

model = lgb.Booster(model_file=MODEL_PATH)
artifacts = load_json(ARTIFACTS_PATH)


def select_features(df):
    selected_cols = [
        "TransactionAmt",
        "TransactionAmt_log",
        "card1",
        "card2",
        "card3",
        "card5",
        "addr1",
        "dist1",
        "card1_freq",
        "card2_freq",
        "addr1_freq",
        "P_emaildomain_freq",
        "card1_amt_mean",
        "card1_amt_std",
        "TransactionDT_days",
        "TransactionDT_hours",
    ]

    for col in selected_cols:
        if col not in df.columns:
            df[col] = 0

    return df[selected_cols]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(transaction: TransactionInput):
    input_df = pd.DataFrame([transaction.model_dump()])

    transformed_df = transform_features(input_df, artifacts)
    X = select_features(transformed_df)

    fraud_probability = float(model.predict(X)[0])

    if fraud_probability >= 0.8:
        risk_level = "high"
    elif fraud_probability >= 0.4:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "transaction_id": transaction.TransactionID,
        "fraud_probability": round(fraud_probability, 6),
        "risk_level": risk_level
    }