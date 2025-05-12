import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report, precision_recall_curve, roc_curve
)

# Ensure project root is on path
root = Path(__file__).parents[2]
sys.path.insert(0, str(root))


def plot_confusion_matrix(cm, labels, title, save_path=None):
    """Plot a confusion matrix with annotations."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curve(y_true, y_probs, model_names, save_path=None):
    """Plot ROC curves for multiple models."""
    plt.figure(figsize=(10, 8))
    
    for i, (prob, name) in enumerate(zip(y_probs, model_names)):
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc = roc_auc_score(y_true, prob)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_precision_recall_curve(y_true, y_probs, model_names, save_path=None):
    """Plot precision-recall curves for multiple models."""
    plt.figure(figsize=(10, 8))
    
    for i, (prob, name) in enumerate(zip(y_probs, model_names)):
        precision, recall, _ = precision_recall_curve(y_true, prob)
        
        # Calculate average precision
        avg_precision = np.mean(precision)
        
        plt.plot(recall, precision, lw=2, 
                 label=f'{name} (Avg Precision = {avg_precision:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def threshold_sensitivity_analysis(y_true, y_prob, model_name, save_path=None):
    thresholds = np.linspace(0.1, 0.9, 41)
    metrics = {
        'threshold': thresholds,
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        metrics['accuracy'].append(accuracy_score(y_true, y_pred))
        metrics['precision'].append(precision_score(y_true, y_pred))
        metrics['recall'].append(recall_score(y_true, y_pred))
        metrics['f1'].append(f1_score(y_true, y_pred))
    
    # Plot threshold sensitivity
    plt.figure(figsize=(12, 8))
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        plt.plot(thresholds, metrics[metric], lw=2, label=metric.capitalize())
    
    plt.xlabel('Decision Threshold')
    plt.ylabel('Metric Value')
    plt.title(f'Threshold Sensitivity Analysis - {model_name}')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    
    # Find and highlight optimal F1 threshold
    optimal_idx = np.argmax(metrics['f1'])
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = metrics['f1'][optimal_idx]
    
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
                label=f'Optimal F1 Threshold: {optimal_threshold:.2f}')
    plt.plot(optimal_threshold, optimal_f1, 'ro', ms=8)
    plt.annotate(f'F1: {optimal_f1:.3f}', 
                 xy=(optimal_threshold, optimal_f1),
                 xytext=(optimal_threshold + 0.05, optimal_f1 - 0.05),
                 arrowprops=dict(arrowstyle='->'))
    plt.legend(loc='best')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return optimal_threshold, metrics


def main():
    # Create necessary directories
    processed_dir = root / "data" / "processed"
    report_dir = root / "reports"
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Enhanced Model Evaluation ===")
    
    # Load balanced test dataset if available, else use processed split
    balanced_fp = processed_dir / "balanced_test.csv"
    if balanced_fp.exists():
        print(f"[DATA] Loading balanced test dataset from {balanced_fp}")
        df_bal = pd.read_csv(balanced_fp)
        X_test = df_bal.drop(columns=["readmitted"]).values
        y_test = df_bal["readmitted"].values
        # Load scaler
        try:
            scaler = joblib.load(processed_dir / "robust_scaler.joblib")
        except FileNotFoundError:
            scaler = joblib.load(processed_dir / "scaler.joblib")
        X_test = scaler.transform(X_test)
        data_source = "Balanced test dataset"
    else:
        # Fallback to processed data split
        print("[DATA] Loading and splitting processed dataset")
        X_df = pd.read_csv(processed_dir / "train_processed.csv")
        raw_df = pd.read_csv(root / "data" / "raw" / "train_df.csv")
        X = X_df.values
        y = raw_df["readmitted"].values
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        try:
            scaler = joblib.load(processed_dir / "robust_scaler.joblib")
        except FileNotFoundError:
            scaler = joblib.load(processed_dir / "scaler.joblib")
        X_test = scaler.transform(X_test)
        data_source = "Processed dataset"

    print(f"Using {data_source} for evaluation")
    print(f"Test set shape: {X_test.shape}")
    print(f"Class distribution in test set: {np.bincount(y_test)}")
    
    # Load models
    try:
        logreg = joblib.load(processed_dir / "logreg_model_best.joblib")
        rf    = joblib.load(processed_dir / "rf_model_best.joblib")
        xgb   = joblib.load(processed_dir / "xgboost_model_best.joblib")
        print("Models loaded successfully: LR, RF, XGB")
    except FileNotFoundError:
        print("Error: Best model files not found. Please run train_best_models.py first.")
        return
    
    # Get model thresholds (if saved)
    try:
        thresholds = joblib.load(processed_dir / "optimal_thresholds.joblib")
        lr_threshold  = thresholds.get('lr', 0.5)
        rf_threshold  = thresholds.get('rf', 0.5)
        xgb_threshold = thresholds.get('xgb', 0.5)
        print(f"Using saved thresholds: LR={lr_threshold:.2f}, RF={rf_threshold:.2f}, XGB={xgb_threshold:.2f}")
    except FileNotFoundError:
        lr_threshold = rf_threshold = xgb_threshold = 0.5
        print("Using default threshold of 0.5 for all models (optimized thresholds not found)")
    
    # Get predictions
    probs_lr  = logreg.predict_proba(X_test)
    preds_lr  = (probs_lr  >= lr_threshold).astype(int)
    
    probs_rf  = rf.predict_proba(X_test)
    preds_rf  = (probs_rf  >= rf_threshold).astype(int)
    
    probs_xgb = xgb.predict_proba(X_test)[:, 1]
    preds_xgb = (probs_xgb >= xgb_threshold).astype(int)
    
    # Threshold sensitivity analysis
    print("\nPerforming threshold sensitivity analysis...")
    
    lr_optimal_threshold, _  = threshold_sensitivity_analysis(
        y_test, probs_lr,  "Logistic Regression", save_path=figures_dir / "lr_threshold_sensitivity.png"
    )
    rf_optimal_threshold, _  = threshold_sensitivity_analysis(
        y_test, probs_rf,  "Random Forest", save_path=figures_dir / "rf_threshold_sensitivity.png"
    )
    xgb_optimal_threshold, _ = threshold_sensitivity_analysis(
        y_test, probs_xgb, "XGBoost",         save_path=figures_dir / "xgb_threshold_sensitivity.png"
    )
    
    # Update predictions with optimal thresholds
    preds_lr_optimal  = (probs_lr  >= lr_optimal_threshold).astype(int)
    preds_rf_optimal  = (probs_rf  >= rf_optimal_threshold).astype(int)
    preds_xgb_optimal = (probs_xgb >= xgb_optimal_threshold).astype(int)
    
    # Save optimal thresholds
    joblib.dump({
        'lr':  lr_optimal_threshold,
        'rf':  rf_optimal_threshold,
        'xgb': xgb_optimal_threshold
    }, processed_dir / "optimal_thresholds.joblib")
    print(f"Optimal thresholds: LR={lr_optimal_threshold:.2f}, RF={rf_optimal_threshold:.2f}, XGB={xgb_optimal_threshold:.2f}")
    
    # Compute metrics with original thresholds
    print("\nMetrics with original thresholds:")
    
    # Logistic Regression metrics
    acc_lr = accuracy_score(y_test, preds_lr)
    f1_lr = f1_score(y_test, preds_lr)
    precision_lr = precision_score(y_test, preds_lr)
    recall_lr = recall_score(y_test, preds_lr)
    auc_lr = roc_auc_score(y_test, probs_lr)
    
    # Random Forest metrics
    acc_rf = accuracy_score(y_test, preds_rf)
    f1_rf = f1_score(y_test, preds_rf)
    precision_rf = precision_score(y_test, preds_rf)
    recall_rf = recall_score(y_test, preds_rf)
    auc_rf = roc_auc_score(y_test, probs_rf)
    
    # XGBoost metrics
    acc_xgb     = accuracy_score(y_test, preds_xgb)
    f1_xgb      = f1_score(y_test, preds_xgb)
    precision_xgb = precision_score(y_test, preds_xgb)
    recall_xgb    = recall_score(y_test, preds_xgb)
    auc_xgb     = roc_auc_score(y_test, probs_xgb)
    
    # Display comprehensive results
    print("\n=== Evaluation with Original Thresholds ===")
    print(f"Logistic Regression (threshold={lr_threshold:.2f}):")
    print(f"  Accuracy:  {acc_lr:.4f}")
    print(f"  F1 Score:  {f1_lr:.4f}")
    print(f"  Precision: {precision_lr:.4f}")
    print(f"  Recall:    {recall_lr:.4f}")
    print(f"  ROC AUC:   {auc_lr:.4f}")
    
    print(f"\nRandom Forest (threshold={rf_threshold:.2f}):")
    print(f"  Accuracy:  {acc_rf:.4f}")
    print(f"  F1 Score:  {f1_rf:.4f}")
    print(f"  Precision: {precision_rf:.4f}")
    print(f"  Recall:    {recall_rf:.4f}")
    print(f"  ROC AUC:   {auc_rf:.4f}")
    
    print(f"\nXGBoost (threshold={xgb_threshold:.2f}):")
    print(f"  Accuracy:  {acc_xgb:.4f}")
    print(f"  F1 Score:  {f1_xgb:.4f}")
    print(f"  Precision: {precision_xgb:.4f}")
    print(f"  Recall:    {recall_xgb:.4f}")
    print(f"  ROC AUC:   {auc_xgb:.4f}")
    
    # Compute metrics with optimal thresholds
    print("\n=== Evaluation with Optimal Thresholds ===")
    
    # Logistic Regression metrics (optimal threshold)
    acc_lr_opt = accuracy_score(y_test, preds_lr_optimal)
    f1_lr_opt = f1_score(y_test, preds_lr_optimal)
    precision_lr_opt = precision_score(y_test, preds_lr_optimal)
    recall_lr_opt = recall_score(y_test, preds_lr_optimal)
    
    # Random Forest metrics (optimal threshold)
    acc_rf_opt = accuracy_score(y_test, preds_rf_optimal)
    f1_rf_opt = f1_score(y_test, preds_rf_optimal)
    precision_rf_opt = precision_score(y_test, preds_rf_optimal)
    recall_rf_opt = recall_score(y_test, preds_rf_optimal)
    
    # XGBoost optimal threshold metrics
    acc_xgb_opt     = accuracy_score(y_test, preds_xgb_optimal)
    f1_xgb_opt      = f1_score(y_test, preds_xgb_optimal)
    precision_xgb_opt = precision_score(y_test, preds_xgb_optimal)
    recall_xgb_opt    = recall_score(y_test, preds_xgb_optimal)
    
    print(f"Logistic Regression (threshold={lr_optimal_threshold:.2f}):")
    print(f"  Accuracy:  {acc_lr_opt:.4f}")
    print(f"  F1 Score:  {f1_lr_opt:.4f}")
    print(f"  Precision: {precision_lr_opt:.4f}")
    print(f"  Recall:    {recall_lr_opt:.4f}")
    
    print(f"\nRandom Forest (threshold={rf_optimal_threshold:.2f}):")
    print(f"  Accuracy:  {acc_rf_opt:.4f}")
    print(f"  F1 Score:  {f1_rf_opt:.4f}")
    print(f"  Precision: {precision_rf_opt:.4f}")
    print(f"  Recall:    {recall_rf_opt:.4f}")
    
    print(f"\nXGBoost (threshold={xgb_optimal_threshold:.2f}):")
    print(f"  Accuracy:  {acc_xgb_opt:.4f}")
    print(f"  F1 Score:  {f1_xgb_opt:.4f}")
    print(f"  Precision: {precision_xgb_opt:.4f}")
    print(f"  Recall:    {recall_xgb_opt:.4f}")
    
    # Generate classification reports
    print("\n=== Detailed Classification Reports ===")
    print("\nLogistic Regression:")
    print(classification_report(y_test, preds_lr_optimal))
    
    print("\nRandom Forest:")
    print(classification_report(y_test, preds_rf_optimal))
    
    print("\nXGBoost:")
    print(classification_report(y_test, preds_xgb_optimal))
    
    # Generate and save visualizations
    print("\nGenerating visualizations...")
    
    # Confusion matrices
    cm_lr = confusion_matrix(y_test, preds_lr_optimal)
    cm_rf = confusion_matrix(y_test, preds_rf_optimal)
    cm_xgb = confusion_matrix(y_test, preds_xgb_optimal)
    
    plot_confusion_matrix(
        cm_lr, ['Not Readmitted', 'Readmitted'], 
        'Logistic Regression Confusion Matrix',
        save_path=figures_dir / "lr_confusion_matrix.png"
    )
    
    plot_confusion_matrix(
        cm_rf, ['Not Readmitted', 'Readmitted'], 
        'Random Forest Confusion Matrix',
        save_path=figures_dir / "rf_confusion_matrix.png"
    )
    
    plot_confusion_matrix(
        cm_xgb, ['Not Readmitted', 'Readmitted'], 
        'XGBoost Confusion Matrix',
        save_path=figures_dir / "xgb_confusion_matrix.png"
    )
    
    # ROC curves
    plot_roc_curve(
        y_test, [probs_lr, probs_rf, probs_xgb], 
        ['Logistic Regression', 'Random Forest', 'XGBoost'],
        save_path=figures_dir / "roc_curves.png"
    )
    
    # Precision-Recall curves
    plot_precision_recall_curve(
        y_test, [probs_lr, probs_rf, probs_xgb], 
        ['Logistic Regression', 'Random Forest', 'XGBoost'],
        save_path=figures_dir / "precision_recall_curves.png"
    )
    
    # Compare to baseline performance (if saved)
    try:
        baseline_metrics = joblib.load(report_dir / "baseline_metrics.joblib")
        
        print("\n=== Performance Improvement Over Baseline ===")
        lr_f1_imp = f1_lr_opt - baseline_metrics.get('lr_f1', 0)
        rf_f1_imp = f1_rf_opt - baseline_metrics.get('rf_f1', 0)
        xgb_f1_imp = f1_xgb_opt - baseline_metrics.get('xgb_f1', 0)
        
        lr_auc_imp = auc_lr - baseline_metrics.get('lr_auc', 0)
        rf_auc_imp = auc_rf - baseline_metrics.get('rf_auc', 0)
        xgb_auc_imp = auc_xgb - baseline_metrics.get('xgb_auc', 0)
        
        print(f"Logistic Regression F1 improvement: {lr_f1_imp:.4f} ({(lr_f1_imp/max(0.0001, baseline_metrics.get('lr_f1', 0)))*100:.1f}%)")
        print(f"Random Forest F1 improvement:        {rf_f1_imp:.4f} ({(rf_f1_imp/max(0.0001, baseline_metrics.get('rf_f1', 0)))*100:.1f}%)")
        print(f"XGBoost F1 improvement:              {xgb_f1_imp:.4f} ({(xgb_f1_imp/max(0.0001, baseline_metrics.get('xgb_f1', 0)))*100:.1f}%)")
        print(f"Logistic Regression AUC improvement: {lr_auc_imp:.4f} ({(lr_auc_imp/max(0.0001, baseline_metrics.get('lr_auc', 0)))*100:.1f}%)")
        print(f"Random Forest AUC improvement:       {rf_auc_imp:.4f} ({(rf_auc_imp/max(0.0001, baseline_metrics.get('rf_auc', 0)))*100:.1f}%)")
        print(f"XGBoost AUC improvement:             {xgb_auc_imp:.4f} ({(xgb_auc_imp/max(0.0001, baseline_metrics.get('xgb_auc', 0)))*100:.1f}%)")
    except FileNotFoundError:
        # Save current metrics as baseline
        joblib.dump({
            'lr_f1': f1_lr_opt,
            'rf_f1': f1_rf_opt,
            'xgb_f1': f1_xgb_opt,
            'lr_auc': auc_lr,
            'rf_auc': auc_rf,
            'xgb_auc': auc_xgb
        }, report_dir / "baseline_metrics.joblib")
        print("\nSaved current metrics as baseline for future comparison")
    
    print(f"\nEvaluation complete. Visualizations saved to {figures_dir}")


if __name__ == "__main__":
    main()