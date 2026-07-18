"""
evaluate.py

Loads the trained model and test set, then produces the evaluation artifacts
that matter for an imbalanced problem: confusion matrix and precision-recall
curve (more informative than ROC-AUC when fraud is ~0.17% of the data).

Usage:
    python src/evaluate.py
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    classification_report,
    average_precision_score,
)

PROCESSED_DIR = "data/processed"
MODEL_PATH = "models/fraud_model.pkl"
FIGURES_DIR = "outputs/figures"


def load_test_data():
    X_test = pd.read_csv(f"{PROCESSED_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{PROCESSED_DIR}/y_test.csv").squeeze()
    return X_test, y_test


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {FIGURES_DIR}/confusion_matrix.png")


def plot_precision_recall_curve(y_test, y_proba):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"Avg Precision = {avg_precision:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/precision_recall_curve.png", dpi=150)
    plt.close()
    print(f"Saved precision-recall curve to {FIGURES_DIR}/precision_recall_curve.png")


def main():
    X_test, y_test = load_test_data()
    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Fraud", "Fraud"]))

    plot_confusion_matrix(y_test, y_pred)
    plot_precision_recall_curve(y_test, y_proba)


if __name__ == "__main__":
    main()
