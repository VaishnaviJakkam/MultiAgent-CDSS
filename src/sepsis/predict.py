from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import DEFAULT_CLASSIFICATION_THRESHOLD, FEATURE_COLUMNS, TARGET

PIPELINE_PATH = Path("models") / "sepsis_pipeline.pkl"
METADATA_PATH = Path("models") / "sepsis_metadata.json"


def _load_pipeline() -> dict[str, Any]:
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Saved Sepsis pipeline not found at {PIPELINE_PATH}")
    pipeline = joblib.load(PIPELINE_PATH)
    if not isinstance(pipeline, dict) or "preprocessor" not in pipeline or "model" not in pipeline:
        raise ValueError("Saved pipeline is not a valid preprocessing + model bundle")
    return pipeline


def _load_threshold() -> float:
    if not METADATA_PATH.exists():
        return DEFAULT_CLASSIFICATION_THRESHOLD
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    threshold = metadata.get("selected_threshold", DEFAULT_CLASSIFICATION_THRESHOLD)
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Saved threshold is outside the valid [0, 1] range")
    return float(threshold)


def predict_sepsis_risk(raw_patient: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
    """Prototype prediction interface for structured patient-level data.

    This accepts a single structured patient record or a DataFrame representing one or more rows.
    A threshold is applied after loading the saved pipeline and is intended for academic prototype use only.
    """
    if isinstance(raw_patient, pd.DataFrame):
        if raw_patient.empty:
            raise ValueError("Input DataFrame is empty")
        frame = raw_patient.copy()
    elif isinstance(raw_patient, dict):
        frame = pd.DataFrame([raw_patient])
    else:
        raise TypeError("Input must be a dict or pandas DataFrame")

    missing = [col for col in FEATURE_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    pipeline = _load_pipeline()
    threshold = _load_threshold()

    transformed = pipeline["preprocessor"].transform(frame[FEATURE_COLUMNS])
    probability = float(pipeline["model"].predict_proba(transformed)[:, 1][0])
    prediction = int(probability >= threshold)

    if probability < threshold:
        risk_level = "LOW"
    elif probability < threshold + 0.15:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return {
        "prediction": prediction,
        "probability": probability,
        "threshold": threshold,
        "risk_level": risk_level,
        "target": TARGET,
    }
