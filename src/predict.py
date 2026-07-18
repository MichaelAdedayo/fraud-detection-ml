"""
predict.py

Loads the saved fraud_model.pkl and scores new transactions in batch.
This is how the .pkl file actually gets used after training: you don't
retrain, you just load it and predict on new data.

Usage:
    python src/predict.py --input data/raw/new_transactions.csv --output outputs/results/scored.csv
"""

import argparse
import pandas as pd
import joblib

MODEL_PATH = "models/fraud_model.pkl"


def score_transactions(input_path: str, output_path: str):
    model = joblib.load(MODEL_PATH)
    new_data = pd.read_csv(input_path)

    # Drop the label column if it's accidentally present
    features = new_data.drop(columns=["Class"], errors="ignore")

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    results = new_data.copy()
    results["fraud_prediction"] = predictions
    results["fraud_probability"] = probabilities

    results.to_csv(output_path, index=False)
    flagged = results["fraud_prediction"].sum()
    print(f"Scored {len(results):,} transactions. Flagged {flagged} as fraud.")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score new transactions for fraud.")
    parser.add_argument("--input", required=True, help="Path to CSV of new transactions")
    parser.add_argument("--output", required=True, help="Path to save scored CSV")
    args = parser.parse_args()

    score_transactions(args.input, args.output)
