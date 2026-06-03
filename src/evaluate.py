"""
Stage 3 — Evaluation
Loads test split, runs inference, outputs metrics.json + CSV plots for DVC/CML.
"""
import json
import os
import yaml
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(params: dict) -> None:
    cfg_data = params["data"]
    cfg_feat = params["features"]
    cfg_eval = params["evaluate"]

    processed_path = cfg_data["processed_path"]
    target_col = cfg_feat["target"]
    test_size = cfg_data["test_size"]
    random_state = cfg_data["random_state"]
    threshold = cfg_eval["threshold"]

    df = pd.read_csv(processed_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = joblib.load("models/model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    scaler = preprocessor["scaler"]

    X_test_scaled = scaler.transform(X_test)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # ── Metrics ──────────────────────────────────────────────────────────────
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }
    print("Metrics:", metrics)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    os.makedirs(cfg_eval["plots_path"], exist_ok=True)
    with open(cfg_eval["metrics_path"], "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {cfg_eval['metrics_path']}")

    # ── Confusion matrix CSV (for DVC plots) ─────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    cm_rows = []
    labels = [0, 1]
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            cm_rows.append({"actual": actual, "predicted": predicted, "count": int(cm[i, j])})
    pd.DataFrame(cm_rows).to_csv("reports/confusion_matrix.csv", index=False)

    # ── ROC curve CSV (for DVC plots) ─────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv("reports/roc_curve.csv", index=False)

    # ── PNG plots (rendered by CML as images in PR comments) ─────────────────
    # Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Normal (0)", "Fault (1)"],
        yticklabels=["Normal (0)", "Fault (1)"],
        xlabel="Predicted", ylabel="Actual",
        title="Confusion Matrix",
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=120)
    plt.close()

    # ROC curve
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("reports/roc_curve.png", dpi=120)
    plt.close()

    # Feature importance (top 15)
    feat_names = preprocessor["feature_names"]
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(15), importances[indices])
    ax.set_xticks(range(15))
    ax.set_xticklabels([feat_names[i] for i in indices], rotation=45, ha="right")
    ax.set(title="Top 15 Feature Importances", ylabel="Importance")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=120)
    plt.close()

    print("Saved plots to reports/")


if __name__ == "__main__":
    params = load_params()
    evaluate(params)
