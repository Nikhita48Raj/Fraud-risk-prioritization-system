from pathlib import Path
import pandas as pd


def load_raw_data(data_dir: str = "data/raw"):
    data_path = Path(data_dir)

    train_transaction = pd.read_csv(data_path / "train_transaction.csv")
    train_identity = pd.read_csv(data_path / "train_identity.csv")
    test_transaction = pd.read_csv(data_path / "test_transaction.csv")
    test_identity = pd.read_csv(data_path / "test_identity.csv")

    train_df = train_transaction.merge(train_identity, on="TransactionID", how="left")
    test_df = test_transaction.merge(test_identity, on="TransactionID", how="left")

    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = load_raw_data()
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)