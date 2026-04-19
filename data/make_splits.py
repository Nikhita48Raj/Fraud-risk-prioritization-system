from pathlib import Path
import pandas as pd


def load_data():
    data_path = Path("data/raw")

    train_transaction = pd.read_csv(data_path / "train_transaction.csv")
    train_identity = pd.read_csv(data_path / "train_identity.csv")

    train_df = train_transaction.merge(
        train_identity,
        on="TransactionID",
        how="left"
    )

    return train_df


def create_time_split(df, validation_ratio=0.2):
    
    df = df.sort_values("TransactionDT")

    split_index = int(len(df) * (1 - validation_ratio))

    train_df = df.iloc[:split_index]
    valid_df = df.iloc[split_index:]

    return train_df, valid_df


if __name__ == "__main__":

    df = load_data()

    train_df, valid_df = create_time_split(df)

    print("Train shape:", train_df.shape)
    print("Valid shape:", valid_df.shape)

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    train_df.to_parquet("data/processed/train.parquet")
    valid_df.to_parquet("data/processed/valid.parquet")

    print("Saved processed splits.")