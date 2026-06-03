"""
Stage 1 — Preprocessing
Loads raw CSV, encodes categoricals, splits, and saves processed features.
"""
import os
import yaml
import pandas as pd
from pathlib import Path


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def preprocess(params: dict) -> None:
    cfg_data = params["data"]
    cfg_feat = params["features"]

    raw_path = cfg_data["raw_path"]
    out_path = cfg_data["processed_path"]

    df = pd.read_csv(raw_path, encoding="utf-8-sig")
    print(f"Loaded {len(df)} rows from {raw_path}")

    # Drop identifier columns
    df = df.drop(columns=cfg_feat["drop_cols"], errors="ignore")

    # One-hot encode categorical features
    cat_cols = cfg_feat.get("categorical_cols", [])
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # Ensure boolean dummies are int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    os.makedirs(Path(out_path).parent, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved processed data to {out_path} — shape: {df.shape}")


if __name__ == "__main__":
    params = load_params()
    preprocess(params)
