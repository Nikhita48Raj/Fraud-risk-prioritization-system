import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from pathlib import Path

from src.features.build_features import fit_feature_artifacts, transform_features


def load_data():
    train_df = pd.read_parquet("data/processed/train.parquet")
    valid_df = pd.read_parquet("data/processed/valid.parquet")
    return train_df, valid_df


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

    selected_cols = [c for c in selected_cols if c in df.columns]
    return df[selected_cols]


if __name__ == "__main__":

    Path("outputs/plots").mkdir(parents=True, exist_ok=True)

    train_df, valid_df = load_data()

    artifacts = fit_feature_artifacts(train_df)

    valid_df = transform_features(valid_df, artifacts)

    X_valid = select_features(valid_df)
    y_valid = valid_df["isFraud"]

    model = lgb.Booster(model_file="outputs/models/lgbm_model.txt")

    probs = model.predict(X_valid)

    frac_pos, mean_pred = calibration_curve(y_valid, probs, n_bins=10)

    plt.figure()
    plt.plot(mean_pred, frac_pos)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Probability")
    plt.title("Calibration Curve")

    plt.savefig("outputs/plots/calibration_curve.png")
    plt.close()

    print("Calibration curve saved")