import json
from pathlib import Path

import pandas as pd
import lightgbm as lgb

from sklearn.metrics import roc_auc_score, average_precision_score

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


def save_feature_importance(model, feature_names):
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    importance_df.to_csv("outputs/reports/lgbm_feature_importance.csv", index=False)

    print("\nTop Feature Importances:")
    print(importance_df.head(10).to_string(index=False))


if __name__ == "__main__":
    train_df, valid_df = load_data()

    artifacts = fit_feature_artifacts(train_df)

    train_df = transform_features(train_df, artifacts)
    valid_df = transform_features(valid_df, artifacts)

    X_train = select_features(train_df)
    X_valid = select_features(valid_df)

    y_train = train_df["isFraud"]
    y_valid = valid_df["isFraud"]

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc"
    )

    preds = model.predict_proba(X_valid)[:, 1]

    roc_auc = roc_auc_score(y_valid, preds)
    pr_auc = average_precision_score(y_valid, preds)

    print("\nLightGBM Results")
    print("ROC-AUC:", roc_auc)
    print("PR-AUC:", pr_auc)

    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)

    model.booster_.save_model("outputs/models/lgbm_model.txt")

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "n_features": X_train.shape[1]
    }

    with open("outputs/reports/lgbm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    save_feature_importance(model, X_train.columns.tolist())