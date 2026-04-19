from pathlib import Path

import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

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
    train_df, valid_df = load_data()

    artifacts = fit_feature_artifacts(train_df)

    train_df = transform_features(train_df, artifacts)
    valid_df = transform_features(valid_df, artifacts)

    X_valid = select_features(valid_df)

    model = lgb.Booster(model_file="outputs/models/lgbm_model.txt")

    Path("outputs/plots").mkdir(parents=True, exist_ok=True)
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)

    sample_size = min(1000, len(X_valid))
    X_sample = X_valid.sample(sample_size, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig("outputs/plots/shap_summary.png", bbox_inches="tight")
    plt.close()

    # mean absolute SHAP importance
    if isinstance(shap_values, list):
        shap_array = shap_values[1]
    else:
        shap_array = shap_values

    mean_abs_shap = abs(shap_array).mean(axis=0)

    shap_importance_df = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    shap_importance_df.to_csv("outputs/reports/shap_feature_importance.csv", index=False)

    print("\nTop SHAP Features:")
    print(shap_importance_df.head(10).to_string(index=False))
    print("\nSaved:")
    print("- outputs/plots/shap_summary.png")
    print("- outputs/reports/shap_feature_importance.csv")