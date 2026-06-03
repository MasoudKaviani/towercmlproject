"""
Stage 2 — Training
Trains a Random Forest on the processed features and saves the model artifacts.
"""
import os
import yaml
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train(params: dict) -> None:
    cfg_data = params["data"]
    cfg_feat = params["features"]
    cfg_train = params["train"]

    processed_path = cfg_data["processed_path"]
    target_col = cfg_feat["target"]
    test_size = cfg_data["test_size"]
    random_state = cfg_data["random_state"]

    df = pd.read_csv(processed_path)
    print(f"Loaded processed data — shape: {df.shape}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Class distribution (train): {dict(y_train.value_counts())}")

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=cfg_train["n_estimators"],
        max_depth=cfg_train["max_depth"],
        min_samples_split=cfg_train["min_samples_split"],
        min_samples_leaf=cfg_train["min_samples_leaf"],
        class_weight=cfg_train["class_weight"],
        random_state=cfg_train["random_state"],
        n_jobs=cfg_train["n_jobs"],
    )
    model.fit(X_train_scaled, y_train)
    print("Model training complete.")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    joblib.dump(
        {"scaler": scaler, "feature_names": list(X.columns)},
        "models/preprocessor.pkl",
    )
    print("Saved models/model.pkl and models/preprocessor.pkl")


if __name__ == "__main__":
    params = load_params()
    train(params)
