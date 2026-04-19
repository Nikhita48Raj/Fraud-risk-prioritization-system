import json
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import average_precision_score, roc_auc_score

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


def precision_at_k(y_true, scores, k):

    df = pd.DataFrame({
        "label": y_true,
        "score": scores
    })

    df_sorted = df.sort_values("score", ascending=False)

    top_k = df_sorted.head(k)

    precision = top_k["label"].mean()

    return precision


def recall_at_k(y_true, scores, k):

    df = pd.DataFrame({
        "label": y_true,
        "score": scores
    })

    total_fraud = df["label"].sum()

    df_sorted = df.sort_values("score", ascending=False)

    top_k = df_sorted.head(k)

    detected_fraud = top_k["label"].sum()

    recall = detected_fraud / total_fraud

    return recall


if __name__ == "__main__":

    train_df, valid_df = load_data()

    artifacts = fit_feature_artifacts(train_df)

    train_df = transform_features(train_df, artifacts)
    valid_df = transform_features(valid_df, artifacts)

    X_train = select_features(train_df)
    X_valid = select_features(valid_df)

    y_train = train_df["isFraud"]
    y_valid = valid_df["isFraud"]

    model = lgb.Booster(model_file="outputs/models/lgbm_model.txt")

    preds = model.predict(X_valid)

    results = {}

    for k in [50, 100, 200, 500]:

        results[f"precision@{k}"] = precision_at_k(y_valid, preds, k)
        results[f"recall@{k}"] = recall_at_k(y_valid, preds, k)

    print("\nRanking Metrics")

    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    with open("outputs/reports/ranking_metrics.json", "w") as f:
        json.dump(results, f, indent=2)