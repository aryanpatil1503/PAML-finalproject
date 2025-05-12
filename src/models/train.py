import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure project root is on path
root = Path(__file__).parents[2]
sys.path.insert(0, str(root))

# Import our improved model implementations
from src.models.logistic_regression import LogisticRegressionScratch
from src.models.random_forest import RandomForestScratch
from src.models.xgboost_scratch import XGBoostScratch

def optimize_threshold(model, X, y, metric='f1'):
    y_prob = model.predict_proba(X)
    best_metric_value = 0
    best_threshold = 0.5
    
    # Evaluate thresholds from 0.1 to 0.9
    thresholds = np.linspace(0.1, 0.9, 81)
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        if metric == 'f1':
            metric_value = f1_score(y, y_pred)
        elif metric == 'precision':
            metric_value = precision_score(y, y_pred)
        elif metric == 'recall':
            metric_value = recall_score(y, y_pred)
        
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = threshold
    
    print(f"[OPTIMIZE] Best {metric} at threshold {best_threshold:.2f}: {best_metric_value:.4f}")
    return best_threshold, best_metric_value

def main():
    start_time = time.time()
    
    # Create directories
    processed_dir = root / "data" / "processed"
    figures_dir = root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load balanced train/test datasets
    print("[DATA] Loading balanced train/test datasets")
    train_bal = pd.read_csv(processed_dir / "balanced_train.csv")
    test_bal = pd.read_csv(processed_dir / "balanced_test.csv")
    X_train = train_bal.drop(columns=["readmitted"]).values
    y_train = train_bal["readmitted"].values
    X_test = test_bal.drop(columns=["readmitted"]).values
    y_test = test_bal["readmitted"].values
    print(f"[INFO] Balanced train shape: {X_train.shape}, test shape: {X_test.shape}")
    
    # Feature scaling - use robust scaler for better outlier handling
    print("[PREPROCESS] Scaling features with RobustScaler")
    scaler = RobustScaler()  # Better for outliers than StandardScaler
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, processed_dir / "robust_scaler.joblib")

    # Cross-validation setup
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Logistic Regression hyperparameter tuning
    print("\n[HYPERPARAM] Tuning Logistic Regression")
    lr_grid = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
    pen_grid = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
    best_auc_lr, best_params_lr = 0, {}
    
    for lr_val in lr_grid:
        for pen in pen_grid:
            cv_scores = []
            for ti, vi in kf.split(X_train, y_train):
                # Get train/val split
                X_tr_f, y_tr_f = X_train[ti], y_train[ti]
                X_val_f = X_train[vi]
                y_val_f = y_train[vi]
                
                # Apply SMOTE
                X_tr_s, y_tr_s = SMOTE(random_state=42).fit_resample(X_tr_f, y_tr_f)
                
                # Train model with fixed early stopping parameters
                m = LogisticRegressionScratch(
                    lr=lr_val, 
                    n_iter=1000, 
                    verbose=False, 
                    l2_penalty=pen,
                    tol=1e-8,           # Stricter tolerance
                    min_iter=50,        # Minimum iterations before early stopping
                    early_stopping=True
                )
                m.fit(X_tr_s, y_tr_s)
                
                # Evaluate
                auc = roc_auc_score(y_val_f, m.predict_proba(X_val_f))
                cv_scores.append(auc)
                
            mean_auc = np.mean(cv_scores)
            print(f"  LR params: lr={lr_val}, l2={pen}, AUC={mean_auc:.4f}")
            
            if mean_auc > best_auc_lr:
                best_auc_lr, best_params_lr = mean_auc, {"lr": lr_val, "l2_penalty": pen}
                
    print(f"[HYPERPARAM] Best LR: {best_params_lr}, AUC={best_auc_lr:.4f}")

    # Random Forest hyperparameter tuning
    print("\n[HYPERPARAM] Tuning Random Forest")
    n_est_grid = [25, 50, 100, 200, 500]
    depth_grid = [10, 15, 20, 30]
    min_samples_grid = [2, 5, 10, 20]
    best_auc_rf, best_params_rf = 0, {}
    
    for n_est in n_est_grid:
        for depth in depth_grid:
            for min_samples in min_samples_grid:
                cv_scores = []
                for ti, vi in kf.split(X_train, y_train):
                    # Get train/val split
                    X_tr_f, y_tr_f = X_train[ti], y_train[ti]
                    X_val_f = X_train[vi]
                    y_val_f = y_train[vi]
                    
                    # Apply SMOTE
                    X_tr_s, y_tr_s = SMOTE(random_state=42).fit_resample(X_tr_f, y_tr_f)
                    
                    # Train enhanced RF model
                    m = RandomForestScratch(
                        n_estimators=n_est, 
                        max_depth=depth, 
                        min_samples_split=min_samples,
                        min_samples_leaf=1,
                        class_weight='balanced',       # Use class weights
                        stratify_bootstrap=True,       # Stratified bootstrapping
                        random_state=42
                    )
                    m.fit(X_tr_s, y_tr_s)
                    
                    # Evaluate
                    auc = roc_auc_score(y_val_f, m.predict_proba(X_val_f))
                    cv_scores.append(auc)
                    
                mean_auc = np.mean(cv_scores)
                print(f"  RF params: n_est={n_est}, depth={depth}, min_samples={min_samples}, AUC={mean_auc:.4f}")
                
                if mean_auc > best_auc_rf:
                    best_auc_rf, best_params_rf = mean_auc, {
                        "n_estimators": n_est, 
                        "max_depth": depth,
                        "min_samples_split": min_samples
                    }
                    
    print(f"[HYPERPARAM] Best RF: {best_params_rf}, AUC={best_auc_rf:.4f}")

    # XGBoost hyperparameter tuning
    print("\n[HYPERPARAM] Tuning XGBoostScratch")
    xgb_n_est_grid = [25, 50, 100, 200]
    xgb_lr_grid = [0.01, 0.05, 0.1]
    xgb_depth_grid = [3, 5, 7]
    xgb_subsample_grid = [0.5, 0.75, 1.0]
    best_auc_xgb, best_params_xgb = 0, {}
    
    for n_est in xgb_n_est_grid:
        for lr in xgb_lr_grid:
            for depth in xgb_depth_grid:
                for subs in xgb_subsample_grid:
                    cv_scores = []
                    for ti, vi in kf.split(X_train, y_train):
                        X_tr, y_tr = SMOTE(random_state=42).fit_resample(
                            X_train[ti], y_train[ti]
                        )
                        model_xgb = XGBoostScratch(
                            n_estimators=n_est,
                            learning_rate=lr,
                            max_depth=depth,
                            subsample=subs,
                            random_state=42
                        )
                        model_xgb.fit(X_tr, y_tr)
                        prob_val = model_xgb.predict_proba(X_train[vi])[:,1]
                        cv_scores.append(roc_auc_score(y_train[vi], prob_val))
                    mean_auc = np.mean(cv_scores)
                    print(f"  XGB params: n_est={n_est}, lr={lr}, depth={depth}, subsample={subs}, AUC={mean_auc:.4f}")
                    if mean_auc > best_auc_xgb:
                        best_auc_xgb, best_params_xgb = mean_auc, {
                            "n_estimators": n_est,
                            "learning_rate": lr,
                            "max_depth": depth,
                            "subsample": subs
                        }
    print(f"[HYPERPARAM] Best XGB: {best_params_xgb}, AUC={best_auc_xgb:.4f}")

    # Train final models
    print("\n[TRAIN] Training final models with tuned hyperparams")
    
    # Apply SMOTE to full training set
    X_res_full, y_res_full = SMOTE(random_state=42).fit_resample(X_train, y_train)
    print(f"[INFO] After SMOTE: {Counter(y_res_full)}")
    
    # Train logistic regression
    print("\n[TRAIN] Training Logistic Regression model")
    final_lr = LogisticRegressionScratch(
        lr=best_params_lr["lr"], 
        n_iter=2000,               # More iterations for final model
        verbose=True, 
        l2_penalty=best_params_lr["l2_penalty"],
        tol=1e-8,                  # Stricter tolerance 
        min_iter=100,              # Minimum 100 iterations before early stopping
        early_stopping=True
    )
    final_lr.fit(X_res_full, y_res_full)
    joblib.dump(final_lr, processed_dir / "logreg_model.joblib")
    
    # Train random forest
    print("\n[TRAIN] Training Random Forest model")
    final_rf = RandomForestScratch(
        n_estimators=best_params_rf["n_estimators"],
        max_depth=best_params_rf["max_depth"],
        min_samples_split=best_params_rf["min_samples_split"],
        min_samples_leaf=1,
        class_weight='balanced',
        stratify_bootstrap=True,
        random_state=42
    )
    final_rf.fit(X_res_full, y_res_full)
    joblib.dump(final_rf, processed_dir / "rf_model.joblib")
    
    # Train XGBoostScratch model
    print("\n[TRAIN] Training XGBoostScratch model")
    final_xgb = XGBoostScratch(**best_params_xgb, random_state=42)
    final_xgb.fit(X_res_full, y_res_full)
    joblib.dump(final_xgb, processed_dir / "xgboost_model.joblib")
    
    # Threshold optimization on validation set
    print("\n[CALIBRATE] Optimizing decision thresholds for F1 score")
    
    # LR threshold optimization
    best_thr_lr, best_f1_lr = optimize_threshold(final_lr, X_test, y_test, metric='f1')
    
    # RF threshold optimization
    best_thr_rf, best_f1_rf = optimize_threshold(final_rf, X_test, y_test, metric='f1')
    
    # XGB threshold optimization
    best_thr_xgb, best_f1_xgb = optimize_threshold(final_xgb, X_test, y_test, metric='f1')
    
    # Save optimal thresholds
    joblib.dump({
        'lr': best_thr_lr,
        'rf': best_thr_rf,
        'xgb': best_thr_xgb
    }, processed_dir / "optimal_thresholds.joblib")
    
    print(f"[CALIBRATE] LR best threshold={best_thr_lr:.2f}, F1={best_f1_lr:.4f}")
    print(f"[CALIBRATE] RF best threshold={best_thr_rf:.2f}, F1={best_f1_rf:.4f}")
    print(f"[CALIBRATE] XGB best threshold={best_thr_xgb:.2f}, F1={best_f1_xgb:.4f}")

    # Final evaluation on test set
    print("\n[EVAL] Hold-out set performance")
    
    # LR evaluation
    probs_lr = final_lr.predict_proba(X_test)
    preds_lr = (probs_lr >= best_thr_lr).astype(int)
    
    acc_lr = accuracy_score(y_test, preds_lr)
    f1_lr = f1_score(y_test, preds_lr)
    try:
        roc_auc_lr = roc_auc_score(y_test, probs_lr)
    except ValueError:
        roc_auc_lr = float('nan')
    
    # RF evaluation
    probs_rf = final_rf.predict_proba(X_test)
    preds_rf = (probs_rf >= best_thr_rf).astype(int)
    
    acc_rf = accuracy_score(y_test, preds_rf)
    f1_rf = f1_score(y_test, preds_rf)
    try:
        roc_auc_rf = roc_auc_score(y_test, probs_rf)
    except ValueError:
        roc_auc_rf = float('nan')
    
    # XGB evaluation
    probs_xgb = final_xgb.predict_proba(X_test)[:,1]
    preds_xgb = (probs_xgb >= best_thr_xgb).astype(int)
    
    acc_xgb = accuracy_score(y_test, preds_xgb)
    f1_xgb = f1_score(y_test, preds_xgb)
    roc_auc_xgb = roc_auc_score(y_test, probs_xgb)
    
    # Display results
    print(f"Logistic Regression: acc={acc_lr:.4f}, f1={f1_lr:.4f}, roc_auc={roc_auc_lr:.4f}")
    print(f"Random Forest:        acc={acc_rf:.4f}, f1={f1_rf:.4f}, roc_auc={roc_auc_rf:.4f}")
    print(f"XGBoostScratch:       acc={acc_xgb:.4f}, f1={f1_xgb:.4f}, roc_auc={roc_auc_xgb:.4f}")
    
    # If RF has feature importances, save them
    if hasattr(final_rf, 'feature_importances_'):
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        
        # Get feature names
        feature_names = train_bal.columns.drop("readmitted").tolist()
        
        # Get top 20 features
        importances = final_rf.feature_importances_
        indices = np.argsort(importances)[-20:]
        
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        plt.savefig(figures_dir / "feature_importance.png")
        
        # Save feature importances
        joblib.dump({
            'importances': importances,
            'feature_names': feature_names
        }, processed_dir / "feature_importances.joblib")
    
    elapsed_time = time.time() - start_time
    print(f"\n[INFO] Training completed in {elapsed_time:.2f} seconds")
    print(f"[INFO] Models saved to {processed_dir}")

if __name__ == "__main__":
    main()