import pandas as pd
import numpy as np

from scipy.stats import wasserstein_distance


def compute_drift(train_df, valid_df, features):

    drift_results = []

    for col in features:

        if col not in train_df.columns:
            continue

        train_values = train_df[col].dropna()
        valid_values = valid_df[col].dropna()

        if train_values.nunique() < 2:
            continue

        drift_score = wasserstein_distance(
            train_values,
            valid_values
        )

        drift_results.append({
            "feature": col,
            "drift_score": drift_score,
            "train_mean": train_values.mean(),
            "valid_mean": valid_values.mean()
        })

    drift_df = pd.DataFrame(drift_results)

    drift_df = drift_df.sort_values(
        "drift_score",
        ascending=False
    )

    return drift_df


if __name__ == "__main__":

    train_df = pd.read_parquet("data/processed/train.parquet")
    valid_df = pd.read_parquet("data/processed/valid.parquet")

    features = [
        "TransactionAmt",
        "card1",
        "addr1",
        "dist1"
    ]

    drift_df = compute_drift(train_df, valid_df, features)

    print("\nDrift Summary:")
    print(drift_df.head(10))