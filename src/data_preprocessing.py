"""
data_preprocessing.py

Loads the raw credit card transaction data, scales the Amount/Time features,
and creates a stratified train/test split (stratified so both sets keep the
same fraud ratio as the original data).

Usage:
    python src/data_preprocessing.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_DATA_PATH = "data/raw/creditcard.csv"
PROCESSED_DIR = "data/processed"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} transactions, {df['Class'].sum()} of which are fraud "
          f"({df['Class'].mean() * 100:.3f}%)")
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """V1-V28 are already PCA-transformed/scaled. Only Amount and Time need scaling."""
    scaler = StandardScaler()
    df = df.copy()
    df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])
    return df


def split_data(df: pd.DataFrame):
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # stratify=y keeps the fraud ratio identical in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {len(X_train):,} rows | fraud rate: {y_train.mean() * 100:.3f}%")
    print(f"Test set:  {len(X_test):,} rows | fraud rate: {y_test.mean() * 100:.3f}%")
    return X_train, X_test, y_train, y_test


def main():
    df = load_data(RAW_DATA_PATH)
    df = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    X_train.to_csv(f"{PROCESSED_DIR}/X_train.csv", index=False)
    X_test.to_csv(f"{PROCESSED_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{PROCESSED_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{PROCESSED_DIR}/y_test.csv", index=False)
    print(f"\nSaved processed train/test splits to {PROCESSED_DIR}/")


if __name__ == "__main__":
    main()
