import pandas as pd
import numpy as np


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TransactionDT_days"] = df["TransactionDT"] // (24 * 60 * 60)
    df["TransactionDT_hours"] = df["TransactionDT"] // (60 * 60)

    return df


def create_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    return df


def fit_feature_artifacts(train_df: pd.DataFrame) -> dict:
    artifacts = {}

    freq_cols = ["card1", "card2", "addr1", "P_emaildomain"]
    freq_cols = [c for c in freq_cols if c in train_df.columns]

    freq_maps = {}
    for col in freq_cols:
        freq_maps[col] = train_df[col].value_counts(dropna=False).to_dict()

    artifacts["freq_cols"] = freq_cols
    artifacts["freq_maps"] = freq_maps

    group_maps = {}

    if "card1" in train_df.columns:
        grp = train_df.groupby("card1")["TransactionAmt"]
        group_maps["card1_amt_mean"] = grp.mean().to_dict()
        group_maps["card1_amt_std"] = grp.std().to_dict()

    artifacts["group_maps"] = group_maps

    return artifacts


def transform_features(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    df = df.copy()

    df = create_time_features(df)
    df = create_amount_features(df)

    for col in artifacts.get("freq_cols", []):
        mapping = artifacts["freq_maps"][col]
        df[f"{col}_freq"] = df[col].map(mapping).fillna(0)

    group_maps = artifacts.get("group_maps", {})

    if "card1_amt_mean" in group_maps and "card1" in df.columns:
        df["card1_amt_mean"] = df["card1"].map(group_maps["card1_amt_mean"])

    if "card1_amt_std" in group_maps and "card1" in df.columns:
        df["card1_amt_std"] = df["card1"].map(group_maps["card1_amt_std"])

    return df