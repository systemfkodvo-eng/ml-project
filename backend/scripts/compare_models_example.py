"""
Example script to compare RandomForestClassifier and LogisticRegression
on the provided sample breast cancer dataset.

This script is intended as a runnable verification harness. It prints
confusion matrices and common classification metrics for both models.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def load_data():
    base = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base, "sample_data", "breast_cancer_sample.csv")
    df = pd.read_csv(path)
    return df


def prepare_xy(df):
    # User can adjust target/feature selection as needed.
    if "target" in df.columns:
        y = df["target"]
        X = df.drop(columns=["target"])
    else:
        # Fallback: assume last column is target
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
    return X, y


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"--- {name} ---")
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}\n")
    return {"confusion_matrix": cm, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    df = load_data()
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Logistic Regression (with simple L2 regularization)
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(X_train, y_train)

    # Evaluate
    results = {}
    results["RandomForest"] = evaluate_model("Random Forest", rf, X_test, y_test)
    results["LogisticRegression"] = evaluate_model("Logistic Regression", lr, X_test, y_test)

    # Simple summary
    print("Summary (Accuracy, Precision, Recall, F1):")
    for name, res in results.items():
        print(name, f"{res['accuracy']:.4f}", f"{res['precision']:.4f}", f"{res['recall']:.4f}", f"{res['f1']:.4f}")


if __name__ == "__main__":
    main()
