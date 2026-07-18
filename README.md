# Fintech Transaction Fraud Detection

A machine learning project that detects fraudulent credit card transactions in a highly
imbalanced dataset (~0.17% fraud). Built to demonstrate an end-to-end data science workflow:
EDA, preprocessing, handling class imbalance, model comparison, and business-oriented evaluation.

## Problem Statement

Fraudulent transactions are extremely rare compared to legitimate ones, which makes this a
classic **imbalanced classification** problem. A naive model that predicts "not fraud" every
single time would be 99.8% "accurate" — and completely useless. This project focuses on
correctly identifying fraud (recall) while keeping false alarms manageable (precision), and
frames the results as a business tradeoff rather than just a modeling exercise.

## Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

- 284,807 transactions made by European cardholders over two days
- 492 fraud cases (0.172% of all transactions)
- Features `V1`–`V28` are PCA-transformed (anonymized) for confidentiality
- `Time` and `Amount` are the only untransformed features
- `Class` is the target: `1` = fraud, `0` = legitimate

Download `creditcard.csv` from Kaggle and place it in `data/raw/`.

## Project Structure

```
fraud-detection-project/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                  # place creditcard.csv here (not committed to git)
│   └── processed/            # cleaned/split data saved here by scripts
├── notebooks/
│   └── fraud_detection_analysis.ipynb   # main walkthrough notebook
├── src/
│   ├── data_preprocessing.py # loads, cleans, scales, splits data
│   ├── train_model.py        # applies SMOTE, trains models
│   ├── evaluate.py           # metrics, confusion matrix, PR curve
│   └── predict.py            # loads saved model, scores new transactions
├── models/
│   └── fraud_model.pkl       # saved trained model (generated after training)
└── outputs/
    ├── figures/               # confusion matrix, PR curve, etc.
    └── results/               # scored predictions CSV
```

## Steps to Run

1. **Set up environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Download the data**
   Download `creditcard.csv` from Kaggle and place it in `data/raw/`.

3. **Preprocess the data**
   ```bash
   python src/data_preprocessing.py
   ```
   Scales `Amount`/`Time`, does a stratified train/test split, saves outputs to `data/processed/`.

4. **Train the models**
   ```bash
   python src/train_model.py
   ```
   Applies SMOTE to the training set, trains Logistic Regression and Random Forest,
   saves the best model to `models/fraud_model.pkl`.

5. **Evaluate**
   ```bash
   python src/evaluate.py
   ```
   Prints precision/recall/F1, saves a confusion matrix and precision-recall curve to
   `outputs/figures/`.

6. **Score new transactions (batch prediction)**
   ```bash
   python src/predict.py --input data/raw/new_transactions.csv --output outputs/results/scored.csv
   ```

7. **(Optional) Walk through the full analysis in the notebook**
   Open `notebooks/fraud_detection_analysis.ipynb` for the narrative version — EDA,
   reasoning, and business interpretation in one place. This is the file to link on your
   portfolio/GitHub since it reads top-to-bottom like a case study.

## Key Results

| Model | Precision | Recall | F1-Score |
|---|---|---|---|
| Logistic Regression | 0.06 | 0.92 | 0.11 |
| Random Forest | 0.81 | 0.80 | —0.80|

**Business takeaway (example template):**
"At the chosen threshold, the model catches 81% of fraudulent transactions while flagging
only 80% of all transactions for manual review — reducing review workload by 80% compared
to reviewing every transaction manually."

## Why Recall/Precision Instead of Accuracy

With fraud at 0.17% of transactions, accuracy is a misleading metric. This project reports:
- **Recall** — of all actual fraud cases, how many did we catch?
- **Precision** — of all cases we flagged as fraud, how many were actually fraud?
- **F1-score** — balance between the two
- **Precision-Recall curve** — shows the tradeoff across thresholds, more informative than
  ROC-AUC for imbalanced problems

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, imbalanced-learn (SMOTE), Matplotlib, Seaborn,
Jupyter Notebook, joblib

## Author

Michael Adedayo — [LinkedIn] · [GitHub]
