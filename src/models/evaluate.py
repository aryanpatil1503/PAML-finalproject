#!/usr/bin/env python3
"""
Evaluate saved Logistic Regression and Random Forest on the hold-out set.
"""
import sys
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ensure project root is on path
root = Path(__file__).parents[2]
sys.path.insert(0, str(root))


def main():
    processed_dir = root / "data" / "processed"

    # load processed train data and raw train target
    X_df = pd.read_csv(processed_dir / "train_processed.csv")
    raw_df = pd.read_csv(root / "data" / "raw" / "train_df.csv")
    X, y = X_df.values, raw_df["readmitted"].values
    # hold-out split (20%) for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # load models
    logreg = joblib.load(processed_dir / "logreg_model.joblib")
    rf = joblib.load(processed_dir / "rf_model.joblib")

    # predictions
    preds_lr = logreg.predict(X_test)
    probs_lr = logreg.predict_proba(X_test)
    preds_rf = rf.predict(X_test)
    probs_rf = rf.predict_proba(X_test)

    # compute metrics
    acc_lr = accuracy_score(y_test, preds_lr)
    f1_lr = f1_score(y_test, preds_lr)
    try:
        roc_auc_lr = roc_auc_score(y_test, probs_lr)
    except ValueError:
        roc_auc_lr = float('nan')

    acc_rf = accuracy_score(y_test, preds_rf)
    f1_rf = f1_score(y_test, preds_rf)
    try:
        roc_auc_rf = roc_auc_score(y_test, probs_rf)
    except ValueError:
        roc_auc_rf = float('nan')

    # display results
    print("Evaluation on Hold-out Set:")
    print(f"Logistic Regression: acc={acc_lr:.4f}, f1={f1_lr:.4f}, roc_auc={roc_auc_lr:.4f}")
    print(f"Random Forest:        acc={acc_rf:.4f}, f1={f1_rf:.4f}, roc_auc={roc_auc_rf:.4f}")


if __name__ == "__main__":
    main()
