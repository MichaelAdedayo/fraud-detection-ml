"""
train_model.py

Applies SMOTE to the training set only (never touch the test set with SMOTE -
that would leak synthetic fraud patterns into your evaluation), then trains
and compares Logistic Regression and Random Forest. Saves the best-performing
model to models/fraud_model.pkl.

Usage:
    python src/train_model.py
"""

import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

PROCESSED_DIR = "data/processed"
MODEL_PATH = "models/fraud_model.pkl"


def load_processed_data():
    X_train = pd.read_csv(f"{PROCESSED_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{PROCESSED_DIR}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE:  {y_resampled.value_counts().to_dict()}")
    return X_resampled, y_resampled


def train_and_compare(X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }

    best_model, best_name, best_f1 = None, None, -1

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=["Not Fraud", "Fraud"]))

        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1:
            best_model, best_name, best_f1 = model, name, f1

    print(f"\nBest model: {best_name} (F1-score: {best_f1:.4f})")
    return best_model, best_name


def main():
    X_train, X_test, y_train, y_test = load_processed_data()
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    best_model, best_name = train_and_compare(X_train_resampled, y_train_resampled, X_test, y_test)

    joblib.dump(best_model, MODEL_PATH)
    print(f"\nSaved best model ({best_name}) to {MODEL_PATH}")


if __name__ == "__main__":
    main()
