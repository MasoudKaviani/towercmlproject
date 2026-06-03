"""
FastAPI serving app for tower_status prediction.
Loads the trained Random Forest model and preprocessor at startup.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Model artifacts ────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "models/preprocessor.pkl")
METRICS_PATH = os.getenv("METRICS_PATH", "reports/metrics.json")

_model = None
_scaler = None
_feature_names: List[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _scaler, _feature_names
    _model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    _scaler = preprocessor["scaler"]
    _feature_names = preprocessor["feature_names"]
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Expected features: {_feature_names}")
    yield
    # cleanup (none needed)


app = FastAPI(
    title="Telecom Tower Status Prediction API",
    description="Predicts whether a telecom tower is faulty (1) or normal (0).",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ────────────────────────────────────────────────────────────────
class TowerFeatures(BaseModel):
    tower_temperature: float = Field(..., example=51.17, description="Tower temperature (°C)")
    input_voltage: float = Field(..., example=204.2, description="Input voltage (V)")
    current_consumption: float = Field(..., example=67.07, description="Current consumption (A)")
    wind_speed: float = Field(..., example=43.19, description="Wind speed (km/h)")
    humidity: float = Field(..., example=55.89, description="Humidity (%)")
    connected_users: int = Field(..., example=1327, description="Number of connected users")
    data_traffic_gb: float = Field(..., example=340.98, description="Data traffic (GB)")
    signal_strength: float = Field(..., example=-85.97, description="Signal strength (dBm)")
    power_outages_24h: int = Field(..., example=5, description="Power outages in last 24h")
    tower_age_years: int = Field(..., example=2, description="Tower age (years)")
    active_antennas: int = Field(..., example=9, description="Number of active antennas")
    backup_battery_charge: float = Field(..., example=95.55, description="Backup battery charge (%)")
    packet_error_rate: float = Field(..., example=7.42, description="Packet error rate (%)")
    days_since_maintenance: int = Field(..., example=135, description="Days since last maintenance")
    tower_type: str = Field(..., example="3G", description="Tower type: 2G, 3G, 4G, 5G")

    class Config:
        json_schema_extra = {
            "example": {
                "tower_temperature": 51.17,
                "input_voltage": 204.2,
                "current_consumption": 67.07,
                "wind_speed": 43.19,
                "humidity": 55.89,
                "connected_users": 1327,
                "data_traffic_gb": 340.98,
                "signal_strength": -85.97,
                "power_outages_24h": 5,
                "tower_age_years": 2,
                "active_antennas": 9,
                "backup_battery_charge": 95.55,
                "packet_error_rate": 7.42,
                "days_since_maintenance": 135,
                "tower_type": "3G",
            }
        }


class PredictionResponse(BaseModel):
    tower_status: int
    status_label: str
    fault_probability: float
    confidence: float


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


# ── Helpers ────────────────────────────────────────────────────────────────
VALID_TOWER_TYPES = ["2G", "3G", "4G", "5G"]


def build_feature_vector(tower: TowerFeatures) -> pd.DataFrame:
    """Convert input to a DataFrame matching the training feature schema."""
    data = tower.model_dump()
    tower_type = data.pop("tower_type").upper()

    row = {k: [v] for k, v in data.items()}

    # Replicate one-hot encoding used in training
    for tt in VALID_TOWER_TYPES:
        row[f"tower_type_{tt}"] = [1 if tower_type == tt else 0]

    df = pd.DataFrame(row)

    # Align to training feature names (fill missing with 0, drop extras)
    for col in _feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[_feature_names]
    return df


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "Tower Status Prediction API"}


@app.get("/health", tags=["health"])
def health():
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "features_count": len(_feature_names),
    }


@app.get("/metrics", tags=["info"])
def metrics():
    """Return the latest evaluation metrics from the last training run."""
    if not Path(METRICS_PATH).exists():
        raise HTTPException(status_code=404, detail="Metrics file not found. Run the pipeline first.")
    with open(METRICS_PATH) as f:
        return json.load(f)


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(tower: TowerFeatures):
    """Predict tower status for a single tower."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    df = build_feature_vector(tower)
    X_scaled = _scaler.transform(df)
    prob = float(_model.predict_proba(X_scaled)[0, 1])
    status = int(prob >= 0.5)

    return PredictionResponse(
        tower_status=status,
        status_label="FAULT" if status == 1 else "NORMAL",
        fault_probability=round(prob, 4),
        confidence=round(max(prob, 1 - prob), 4),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
def predict_batch(towers: List[TowerFeatures]):
    """Predict tower status for a batch of towers (max 500)."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if len(towers) > 500:
        raise HTTPException(status_code=400, detail="Batch size must not exceed 500.")

    frames = [build_feature_vector(t) for t in towers]
    X = pd.concat(frames, ignore_index=True)
    X_scaled = _scaler.transform(X)
    probs = _model.predict_proba(X_scaled)[:, 1]
    statuses = (probs >= 0.5).astype(int)

    results = [
        PredictionResponse(
            tower_status=int(s),
            status_label="FAULT" if s == 1 else "NORMAL",
            fault_probability=round(float(p), 4),
            confidence=round(float(max(p, 1 - p)), 4),
        )
        for s, p in zip(statuses, probs)
    ]
    return BatchPredictionResponse(predictions=results, count=len(results))
