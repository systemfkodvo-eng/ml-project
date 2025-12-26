"""
Evaluate and plot comparison between RandomForest and LogisticRegression.

Outputs:
 - backend/app/ml/models/plots/ : PNG files (ROC, PR, calibration, histograms, feature importance)
 - backend/app/ml/models/metrics_comparison.csv : CSV with holdout and CV metrics
 - backend/app/ml/models/pipelines/ : saved pipelines (joblib)

Run from backend/ with:
    python scripts/evaluate_and_plot.py

Requires: pandas, scikit-learn, matplotlib, seaborn, joblib, numpy
Optional: shap, statsmodels
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
import joblib


def ensure_dirs(base_models_dir):
    plots_dir = os.path.join(base_models_dir, "plots")
    pipes_dir = os.path.join(base_models_dir, "pipelines")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(pipes_dir, exist_ok=True)
    return plots_dir, pipes_dir


def load_data():
    base = os.path.dirname(os.path.dirname(__file__))
    sample_dir = os.path.join(base, "sample_data")
    sample_path = os.path.join(sample_dir, "breast_cancer_sample.csv")
    # Prefer a prepared evaluation file if present
    eval_path = os.path.join(sample_dir, "breast_cancer_for_eval.csv")

    # Try evaluation-specific file first
    for p in (eval_path, sample_path):
        try:
            df = pd.read_csv(p)
            # if file has more than one sample, use it
            if df.shape[0] > 1:
                return df
        except Exception:
            pass

    # Fallback: try to load the full dataset in repo root ("breast cancer.csv")
    root_alt = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'breast cancer.csv')
    try:
        df_root = pd.read_csv(root_alt)
        # If it contains 'diagnosis' map to 'target'
        if 'diagnosis' in df_root.columns:
            df_root = df_root.rename(columns={'diagnosis': 'target'})
            df_root['target'] = df_root['target'].map({'M': 1, 'B': 0})
            if 'id' in df_root.columns:
                df_root = df_root.drop(columns=['id'])
            # Save a copy for future runs
            os.makedirs(sample_dir, exist_ok=True)
            out = os.path.join(sample_dir, 'breast_cancer_for_eval.csv')
            df_root.to_csv(out, index=False)
            return df_root
    except Exception:
        pass

    # Final attempt: try to read whatever sample file exists and return it (even if small)
    try:
        df = pd.read_csv(sample_path)
        return df
    except Exception as e:
        raise RuntimeError(f"No usable sample data found. Tried {eval_path}, {sample_path}, {root_alt}: {e}")


def prepare_X_y(df):
    if "target" in df.columns:
        y = df["target"].values
        X = df.drop(columns=["target"])
    else:
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1]
    return X, y


def build_pipelines():
    # RF minimal preprocessing
    rf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ])

    # LR requires scaling
    lr_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, solver="lbfgs")),
    ])

    return rf_pipe, lr_pipe


def metrics_from_preds(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"tp": int(cm[1, 1]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tn": int(cm[0, 0]),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def evaluate_holdout(rf_pipe, lr_pipe, X_train, X_test, y_train, y_test, plots_dir, pipes_dir, base_name="breast_cancer"):
    # Fit and evaluate on holdout
    rf_pipe.fit(X_train, y_train)
    lr_pipe.fit(X_train, y_train)

    # Save pipelines
    joblib.dump(rf_pipe, os.path.join(pipes_dir, f"{base_name}_rf_pipe.joblib"))
    joblib.dump(lr_pipe, os.path.join(pipes_dir, f"{base_name}_lr_pipe.joblib"))

    y_pred_rf = rf_pipe.predict(X_test)
    y_pred_lr = lr_pipe.predict(X_test)

    res_rf = metrics_from_preds(y_test, y_pred_rf)
    res_lr = metrics_from_preds(y_test, y_pred_lr)

    # Probabilities for curves
    try:
        y_proba_rf = rf_pipe.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba_rf = None
    try:
        y_proba_lr = lr_pipe.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba_lr = None

    # ROC and PR plots
    if y_proba_rf is not None and y_proba_lr is not None:
        plt.figure()
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
        fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
        auc_rf = auc(fpr_rf, tpr_rf)
        auc_lr = auc(fpr_lr, tpr_lr)
        plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC={auc_rf:.3f})")
        plt.plot(fpr_lr, tpr_lr, label=f"LR (AUC={auc_lr:.3f})")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "roc_comparison.png"), bbox_inches="tight")
        plt.close()

        # PR curve
        plt.figure()
        prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_proba_rf)
        prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_proba_lr)
        # compute area under PR with compatibility for numpy 2.x
        try:
            auc_pr_rf = np.trapz(prec_rf, rec_rf)
        except AttributeError:
            auc_pr_rf = np.trapezoid(prec_rf, rec_rf)
        try:
            auc_pr_lr = np.trapz(prec_lr, rec_lr)
        except AttributeError:
            auc_pr_lr = np.trapezoid(prec_lr, rec_lr)
        plt.plot(rec_rf, prec_rf, label=f"RF (AUCpr={auc_pr_rf:.3f})")
        plt.plot(rec_lr, prec_lr, label=f"LR (AUCpr={auc_pr_lr:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall curve")
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "pr_comparison.png"), bbox_inches="tight")
        plt.close()

        # Calibration plots & Brier
        plt.figure(figsize=(6, 6))
        prob_true_rf, prob_pred_rf = calibration_curve(y_test, y_proba_rf, n_bins=10)
        prob_true_lr, prob_pred_lr = calibration_curve(y_test, y_proba_lr, n_bins=10)
        plt.plot(prob_pred_rf, prob_true_rf, marker='o', label=f"RF (Brier={brier_score_loss(y_test, y_proba_rf):.3f})")
        plt.plot(prob_pred_lr, prob_true_lr, marker='o', label=f"LR (Brier={brier_score_loss(y_test, y_proba_lr):.3f})")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed probability")
        plt.title("Calibration curve")
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "calibration_comparison.png"), bbox_inches="tight")
        plt.close()

    # Feature importance for RF (if available)
    try:
        rf_model = rf_pipe.named_steps.get("rf")
        importances = rf_model.feature_importances_
        feat_names = getattr(X_test, "columns", None)
        if feat_names is None:
            try:
                feat_names = [f"f{i}" for i in range(X_test.shape[1])]
            except Exception:
                feat_names = list(range(len(importances)))
        imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(20)
        plt.figure(figsize=(8, 6))
        sns.barplot(x="importance", y="feature", data=imp_df)
        plt.title("Top 20 Feature Importances - RF")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "rf_feature_importance.png"), bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    return res_rf, res_lr, y_pred_rf, y_pred_lr, (y_proba_rf, y_proba_lr)


def stratified_cv_compare(rf_pipe, lr_pipe, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {"rf": [], "lr": []}
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        rf_pipe.fit(X_tr, y_tr)
        lr_pipe.fit(X_tr, y_tr)
        ypr_rf = rf_pipe.predict(X_val)
        ypr_lr = lr_pipe.predict(X_val)
        metrics["rf"].append(metrics_from_preds(y_val, ypr_rf))
        metrics["lr"].append(metrics_from_preds(y_val, ypr_lr))
    return metrics


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    # Build contingency table for discordant errors
    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)
    n01 = int(((~a_correct) & b_correct).sum())  # a wrong, b correct
    n10 = int((a_correct & (~b_correct)).sum())  # a correct, b wrong
    # Try statsmodels exact test
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        table = [[0, n10], [n01, 0]]
        res = mcnemar(table, exact=False)
        return {"n01": n01, "n10": n10, "statistic": res.statistic, "pvalue": res.pvalue}
    except Exception:
        # approximate chi-square
        stat = (abs(n10 - n01) - 1) ** 2 / (n10 + n01 + 1e-9)
        from math import exp
        # p-value approx using chi2 with 1 df
        try:
            from scipy.stats import chi2
            p = 1 - chi2.cdf(stat, df=1)
        except Exception:
            p = np.nan
        return {"n01": n01, "n10": n10, "statistic": stat, "pvalue": p}


def bootstrap_diff_metric(y_true, y_proba_a, y_proba_b, metric_fn, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    diffs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        ya = (y_proba_a[idx] >= 0.5).astype(int)
        yb = (y_proba_b[idx] >= 0.5).astype(int)
        diffs.append(metric_fn(y_true[idx], ya) - metric_fn(y_true[idx], yb))
    diffs = np.array(diffs)
    lo = np.percentile(diffs, 2.5)
    hi = np.percentile(diffs, 97.5)
    return diffs.mean(), lo, hi


def _area_pr(y_prec, y_rec):
    if hasattr(np, "trapz"):
        return np.trapz(y_prec, y_rec)
    return np.trapezoid(y_prec, y_rec)


def save_metrics_csv(out_csv_path, holdout_rf, holdout_lr, cv_metrics):
    rows = []
    rows.append({"model": "RandomForest_holdout", **holdout_rf})
    rows.append({"model": "LogisticRegression_holdout", **holdout_lr})
    # CV
    for i, m in enumerate(cv_metrics["rf"]):
        rows.append({"model": f"RandomForest_cv_fold_{i+1}", **m})
    for i, m in enumerate(cv_metrics["lr"]):
        rows.append({"model": f"LogisticRegression_cv_fold_{i+1}", **m})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv_path, index=False)


def main():
    df = load_data()
    X_df, y = prepare_X_y(df)
    # convert to numpy for consistent slicing
    X = X_df.values

    base_models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "ml", "models")
    plots_dir, pipes_dir = ensure_dirs(base_models_dir)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    rf_pipe, lr_pipe = build_pipelines()

    # Holdout evaluation
    hold_rf, hold_lr, ypr_rf, ypr_lr, (y_prob_rf, y_prob_lr) = evaluate_holdout(rf_pipe, lr_pipe, X_train, X_test, y_train, y_test, plots_dir, pipes_dir)

    # CV
    cv_metrics = stratified_cv_compare(rf_pipe, lr_pipe, X, y, n_splits=5)

    # McNemar on holdout
    mcnemar_res = mcnemar_test(y_test, np.array(ypr_rf), np.array(ypr_lr))

    # Bootstrap difference on recall and f1 (requires probabilities)
    boot_rec = (np.nan, np.nan, np.nan)
    boot_f1 = (np.nan, np.nan, np.nan)
    if (y_prob_rf is not None) and (y_prob_lr is not None):
        boot_rec = bootstrap_diff_metric(y_test, y_prob_rf, y_prob_lr, lambda yt, yp: recall_score(yt, yp, zero_division=0), n_boot=500)
        boot_f1 = bootstrap_diff_metric(y_test, y_prob_rf, y_prob_lr, lambda yt, yp: f1_score(yt, yp, zero_division=0), n_boot=500)

    # Save metrics and CV
    out_csv = os.path.join(base_models_dir, "metrics_comparison.csv")
    save_metrics_csv(out_csv, hold_rf, hold_lr, cv_metrics)

    # Save summary text
    summary_path = os.path.join(base_models_dir, "evaluation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Holdout RandomForest:\n")
        f.write(str(hold_rf) + "\n\n")
        f.write("Holdout LogisticRegression:\n")
        f.write(str(hold_lr) + "\n\n")
        f.write("McNemar result:\n")
        f.write(str(mcnemar_res) + "\n\n")
        f.write(f"Bootstrap Recall diff mean, lo, hi: {boot_rec}\n")
        f.write(f"Bootstrap F1 diff mean, lo, hi: {boot_f1}\n")

    print("Evaluation complete. Artifacts saved to:")
    print(" - metrics CSV:", out_csv)
    print(" - plots:", plots_dir)
    print(" - pipelines:", pipes_dir)
    print(" - summary:", summary_path)


if __name__ == "__main__":
    main()
