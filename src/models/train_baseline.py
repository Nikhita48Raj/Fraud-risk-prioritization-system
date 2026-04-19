import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.features.build_features import fit_feature_artifacts, transform_features


def load_data():
    train_df = pd.read_parquet("data/processed/train.parquet")
    valid_df = pd.read_parquet("data/processed/valid.parquet")
    return train_df, valid_df


def select_features(df):
    selected_cols = [
        "TransactionAmt",
        "TransactionAmt_log",
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

    X_train = select_features(train_df)
    X_valid = select_features(valid_df)

    y_train = train_df["isFraud"]
    y_valid = valid_df["isFraud"]

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=300))
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict_proba(X_valid)[:, 1]

    roc_auc = roc_auc_score(y_valid, preds)
    pr_auc = average_precision_score(y_valid, preds)

    print("ROC-AUC:", roc_auc)
    print("PR-AUC:", pr_auc)