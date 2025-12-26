"""
Compare RandomForestClassifier vs SVM (SVC) and produce a detailed comparison figure.

Saves a single PNG containing:
 - bar chart of Accuracy/Precision/Recall/F1
 - ROC curves with AUC
 - Precision-Recall curves with area
 - Confusion matrices for both models
 - Text summary with McNemar test and bootstrap CI for recall difference

Run from `backend/` with:
    python scripts/compare_rf_vs_svm.py

Requires: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, scipy (optional)
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
from sklearn.svm import SVC
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
import joblib


def load_data():
    base = os.path.dirname(os.path.dirname(__file__))
    sample_dir = os.path.join(base, "sample_data")
    sample_path = os.path.join(sample_dir, "breast_cancer_sample.csv")
    eval_path = os.path.join(sample_dir, "breast_cancer_for_eval.csv")

    for p in (eval_path, sample_path):
        try:
            df = pd.read_csv(p)
            if df.shape[0] > 1:
                return df
        except Exception:
            pass

    root_alt = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'breast cancer.csv')
    try:
        df_root = pd.read_csv(root_alt)
        if 'diagnosis' in df_root.columns:
            df_root = df_root.rename(columns={'diagnosis': 'target'})
            df_root['target'] = df_root['target'].map({'M': 1, 'B': 0})
            if 'id' in df_root.columns:
                df_root = df_root.drop(columns=['id'])
            os.makedirs(sample_dir, exist_ok=True)
            out = os.path.join(sample_dir, 'breast_cancer_for_eval.csv')
            df_root.to_csv(out, index=False)
            return df_root
    except Exception:
        pass

    try:
        df = pd.read_csv(sample_path)
        return df
    except Exception as e:
        raise RuntimeError(f"No usable sample data found: {e}")


def prepare_X_y(df):
    if "target" in df.columns:
        y = df["target"].values
        X = df.drop(columns=["target"]).copy()
    else:
        y = df.iloc[:, -1].values
        X = df.iloc[:, :-1].copy()
    return X, y


def ensure_dirs(base_models_dir):
    plots_dir = os.path.join(base_models_dir, "plots")
    pipes_dir = os.path.join(base_models_dir, "pipelines")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(pipes_dir, exist_ok=True)
    return plots_dir, pipes_dir


def metrics_from_preds(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"tp": int(cm[1, 1]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tn": int(cm[0, 0]),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)
    n01 = int(((~a_correct) & b_correct).sum())
    n10 = int((a_correct & (~b_correct)).sum())
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        table = [[0, n10], [n01, 0]]
        res = mcnemar(table, exact=False)
        return {"n01": n01, "n10": n10, "statistic": float(res.statistic), "pvalue": float(res.pvalue)}
    except Exception:
        stat = (abs(n10 - n01) - 1) ** 2 / (n10 + n01 + 1e-9)
        try:
            from scipy.stats import chi2
            p = 1 - chi2.cdf(stat, df=1)
        except Exception:
            p = np.nan
        return {"n01": n01, "n10": n10, "statistic": float(stat), "pvalue": float(p) if not np.isnan(p) else np.nan}


def _area_pr(y_prec, y_rec):
    # numpy 2.x renamed trapz -> trapezoid; support both
    if hasattr(np, "trapz"):
        return np.trapz(y_prec, y_rec)
    return np.trapezoid(y_prec, y_rec)


def bootstrap_diff_metric(y_true, y_proba_a, y_proba_b, metric_fn, n_boot=500, seed=42):
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


def plot_comparison(metrics_rf, metrics_svm, y_test, y_prob_rf, y_prob_svm, y_pred_rf, y_pred_svm, outpath):
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.8], height_ratios=[1, 1, 1], hspace=0.4, wspace=0.4)

    # Metrics bar chart (top full row)
    ax0 = fig.add_subplot(gs[0, :3])
    metrics_names = ["accuracy", "precision", "recall", "f1"]
    rf_vals = [metrics_rf[m] for m in metrics_names]
    svm_vals = [metrics_svm[m] for m in metrics_names]
    x = np.arange(len(metrics_names))
    width = 0.35
    ax0.bar(x - width/2, rf_vals, width, label="RandomForest")
    ax0.bar(x + width/2, svm_vals, width, label="SVM")
    ax0.set_xticks(x)
    ax0.set_xticklabels([m.capitalize() for m in metrics_names])
    ax0.set_ylim(0, 1)
    ax0.set_title("Comparison of classification metrics")
    ax0.legend()

    # ROC curves (middle left)
    ax1 = fig.add_subplot(gs[1, 0])
    if y_prob_rf is not None and y_prob_svm is not None:
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
        fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
        auc_rf = auc(fpr_rf, tpr_rf)
        auc_svm = auc(fpr_svm, tpr_svm)
        ax1.plot(fpr_rf, tpr_rf, label=f"RF (AUC={auc_rf:.3f})")
        ax1.plot(fpr_svm, tpr_svm, label=f"SVM (AUC={auc_svm:.3f})")
        ax1.plot([0, 1], [0, 1], "--", color="gray")
    ax1.set_title("ROC Curve")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend()

    # PR curves (middle center)
    ax2 = fig.add_subplot(gs[1, 1])
    if y_prob_rf is not None and y_prob_svm is not None:
        prec_rf, rec_rf, _ = precision_recall_curve(y_test, y_prob_rf)
        prec_svm, rec_svm, _ = precision_recall_curve(y_test, y_prob_svm)
        auc_pr_rf = _area_pr(prec_rf, rec_rf)
        auc_pr_svm = _area_pr(prec_svm, rec_svm)
        ax2.plot(rec_rf, prec_rf, label=f"RF (AUCpr={auc_pr_rf:.3f})")
        ax2.plot(rec_svm, prec_svm, label=f"SVM (AUCpr={auc_pr_svm:.3f})")
    ax2.set_title("Precision-Recall Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend()

    # Confusion matrix RF (middle right)
    ax3 = fig.add_subplot(gs[1, 2])
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax3)
    ax3.set_title("RF Confusion Matrix")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")

    # Confusion matrix SVM (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Oranges", cbar=False, ax=ax4)
    ax4.set_title("SVM Confusion Matrix")
    ax4.set_xlabel("Predicted")
    ax4.set_ylabel("Actual")

    # Summary text (right column spanning rows)
    ax5 = fig.add_subplot(gs[1:, 3])
    ax5.axis("off")
    lines = []
    lines.append("Model summary:")
    lines.append("")
    def fmt(m):
        return f"Acc={m['accuracy']:.3f}, Prec={m['precision']:.3f}, Rec={m['recall']:.3f}, F1={m['f1']:.3f}"
    lines.append(f"RandomForest: {fmt(metrics_rf)}")
    lines.append(f"SVM:          {fmt(metrics_svm)}")
    ax5.text(0, 0.9, "\n".join(lines), fontsize=10, family='monospace')

    plt.suptitle("RandomForest vs SVM - Detailed Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    print(f"[compare_rf_vs_svm] saving figure to: {outpath}")
    plt.savefig(outpath, dpi=200)
    print(f"[compare_rf_vs_svm] saved figure: exists={os.path.exists(outpath)} size={os.path.getsize(outpath) if os.path.exists(outpath) else 'N/A'}")
    plt.close()


def main():
    df = load_data()
    X_df, y = prepare_X_y(df)
    X = X_df.values

    base_models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "ml", "models")
    plots_dir, pipes_dir = ensure_dirs(base_models_dir)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Pipelines
    rf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ])

    svm_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", SVC(probability=True, kernel="rbf", C=1.0, random_state=42)),
    ])

    # Fit
    rf_pipe.fit(X_train, y_train)
    svm_pipe.fit(X_train, y_train)

    # Save pipelines
    joblib.dump(rf_pipe, os.path.join(pipes_dir, "rf_pipe.joblib"))
    joblib.dump(svm_pipe, os.path.join(pipes_dir, "svm_pipe.joblib"))

    # Predictions
    y_pred_rf = rf_pipe.predict(X_test)
    y_pred_svm = svm_pipe.predict(X_test)
    try:
        y_prob_rf = rf_pipe.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob_rf = None
    try:
        y_prob_svm = svm_pipe.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob_svm = None

    metrics_rf = metrics_from_preds(y_test, y_pred_rf)
    metrics_svm = metrics_from_preds(y_test, y_pred_svm)

    # Statistical checks
    mcnemar_res = mcnemar_test(y_test, y_pred_rf, y_pred_svm)
    boot_rec = (np.nan, np.nan, np.nan)
    boot_f1 = (np.nan, np.nan, np.nan)
    if (y_prob_rf is not None) and (y_prob_svm is not None):
        boot_rec = bootstrap_diff_metric(y_test, y_prob_rf, y_prob_svm, lambda yt, yp: recall_score(yt, yp, zero_division=0), n_boot=500)
        boot_f1 = bootstrap_diff_metric(y_test, y_prob_rf, y_prob_svm, lambda yt, yp: f1_score(yt, yp, zero_division=0), n_boot=500)

    outpath = os.path.join(plots_dir, "rf_vs_svm_comparison.png")
    plot_comparison(metrics_rf, metrics_svm, y_test, y_prob_rf, y_prob_svm, y_pred_rf, y_pred_svm, outpath)

    # Save summary
    summary = {
        "rf": metrics_rf,
        "svm": metrics_svm,
        "mcnemar": mcnemar_res,
        "bootstrap_recall_diff": boot_rec,
        "bootstrap_f1_diff": boot_f1,
    }
    summary_path = os.path.join(base_models_dir, "rf_vs_svm_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        import json
        json.dump(summary, f, indent=2)

    print("Saved comparison figure:", outpath)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
