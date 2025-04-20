#!/usr/bin/env python3
"""
Train scratch Logistic Regression and Random Forest with cross-validation and save final models.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ensure project root on path
root = Path(__file__).parents[2]
sys.path.insert(0, str(root))

from src.models.logistic_regression import LogisticRegressionScratch
from src.models.random_forest import RandomForestScratch


def main():
    processed_dir = root / "data" / "processed"
    train_path = processed_dir / "train_processed.csv"

    # load processed train data and raw target
    X_df = pd.read_csv(train_path)
    raw_df = pd.read_csv(root / "data" / "raw" / "train_df.csv")
    X = X_df.values
    y = raw_df["readmitted"].values

    # 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accs_lr, f1s_lr, aucs_lr = [], [], []
    accs_rf, f1s_rf, aucs_rf = [], [], []
    print("[TRAIN] Starting 5-fold CV")
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        # Logistic Regression
        model_lr = LogisticRegressionScratch(lr=0.01, n_iter=1000, verbose=False, l2_penalty=0.1)
        model_lr.fit(X_tr, y_tr)
        preds_lr = model_lr.predict(X_val)
        probs_lr = model_lr.predict_proba(X_val)
        acc_lr = accuracy_score(y_val, preds_lr)
        f1_lr = f1_score(y_val, preds_lr)
        auc_lr = roc_auc_score(y_val, probs_lr)
        print(f"[CV] Fold {fold} LR: acc={acc_lr:.4f}, f1={f1_lr:.4f}, roc_auc={auc_lr:.4f}")
        accs_lr.append(acc_lr); f1s_lr.append(f1_lr); aucs_lr.append(auc_lr)
        # Random Forest
        model_rf = RandomForestScratch(n_estimators=10, max_depth=None, min_samples_split=2)
        model_rf.fit(X_tr, y_tr)
        preds_rf = model_rf.predict(X_val)
        probs_rf = model_rf.predict_proba(X_val)
        acc_rf = accuracy_score(y_val, preds_rf)
        f1_rf = f1_score(y_val, preds_rf)
        try:
            auc_rf = roc_auc_score(y_val, probs_rf)
        except ValueError:
            auc_rf = float('nan')
        print(f"[CV] Fold {fold} RF: acc={acc_rf:.4f}, f1={f1_rf:.4f}, roc_auc={auc_rf:.4f}")
        accs_rf.append(acc_rf); f1s_rf.append(f1_rf); aucs_rf.append(auc_rf)

    print(f"[CV LR] Mean: acc={np.mean(accs_lr):.4f}±{np.std(accs_lr):.4f}, "
          f"f1={np.mean(f1s_lr):.4f}±{np.std(f1s_lr):.4f}, roc_auc={np.mean(aucs_lr):.4f}±{np.std(aucs_lr):.4f}")
    print(f"[CV RF] Mean: acc={np.mean(accs_rf):.4f}±{np.std(accs_rf):.4f}, "
          f"f1={np.mean(f1s_rf):.4f}±{np.std(f1s_rf):.4f}, roc_auc={np.nanmean(aucs_rf):.4f}±{np.nanstd(aucs_rf):.4f}")

    # train final models on full data
    print("[TRAIN] Training final Logistic Regression on full dataset")
    final_model = LogisticRegressionScratch(lr=0.01, n_iter=1000, verbose=True, l2_penalty=0.1)
    final_model.fit(X, y)
    # save Logistic Regression
    model_path = processed_dir / "logreg_model.joblib"
    joblib.dump(final_model, model_path)
    print(f"[TRAIN] Saved final LR model to {model_path}")
    # train and save Random Forest on full data
    print("[TRAIN] Training final Random Forest on full dataset")
    final_rf = RandomForestScratch(n_estimators=10, max_depth=None, min_samples_split=2)
    final_rf.fit(X, y)
    rf_path = processed_dir / "rf_model.joblib"
    joblib.dump(final_rf, rf_path)
    print(f"[TRAIN] Saved final RF model to {rf_path}")


if __name__ == "__main__":
    main()
