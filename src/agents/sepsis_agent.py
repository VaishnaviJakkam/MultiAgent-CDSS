from __future__ import annotations

import joblib
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from src.sepsis.config import PATIENT_ID_COLUMN, TIME_COLUMNS, DEFAULT_CLASSIFICATION_THRESHOLD
from src.sepsis.temporal_features import add_temporal_features, sort_patient_observations, TEMPORAL_BASE_COLUMNS


class SepsisDetectionAgent:
    """Sepsis detection decision-support agent (prototype).

    Responsibilities:
    - Validate incoming structured patient history
    - Generate temporal features using the same functions as the training experiment
    - Load the saved temporal ML pipeline and produce a probability + prediction

    Notes:
    - This is a prototype. The agent does NOT diagnose or recommend treatment.
    """

    def __init__(self, model_path: str | Path, threshold: float | None = None, risk_map: Dict[int, str] | None = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        payload = joblib.load(self.model_path)
        if not isinstance(payload, dict):
            raise ValueError("Loaded model artifact has unexpected format; expected a dict payload")

        required_keys = {"preprocessor", "model", "feature_columns"}
        if not required_keys.issubset(set(payload.keys())):
            raise ValueError(f"Model artifact missing required keys: {required_keys - set(payload.keys())}")

        self.pipeline = payload
        self.preprocessor = payload["preprocessor"]
        self.model = payload["model"]
        self.feature_columns: List[str] = list(payload["feature_columns"])

        # Determine threshold: explicit param overrides saved pipeline threshold
        saved_threshold = float(payload.get("threshold", DEFAULT_CLASSIFICATION_THRESHOLD))
        self.threshold = float(saved_threshold if threshold is None else threshold)
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be between 0 and 1")

        # Simple prototype risk mapping
        self.risk_map = risk_map or {1: "ELEVATED", 0: "LOWER"}

    def _validate_history(self, patient_history: Iterable[Dict[str, Any]]) -> pd.DataFrame:
        if patient_history is None:
            raise ValueError("patient_history is required and cannot be None")

        df = pd.DataFrame(list(patient_history))
        if df.empty:
            raise ValueError("patient_history must contain at least one observation")

        if PATIENT_ID_COLUMN not in df.columns:
            raise ValueError(f"Missing required column: {PATIENT_ID_COLUMN}")

        # Ensure all records have the same Patient_ID
        patient_ids = df[PATIENT_ID_COLUMN].dropna().unique()
        if len(patient_ids) == 0:
            raise ValueError(f"No valid {PATIENT_ID_COLUMN} values found")
        if len(patient_ids) > 1:
            raise ValueError("Inconsistent Patient_ID values found in patient_history")

        # Ensure a time column exists for ordering
        if not any(col in df.columns for col in TIME_COLUMNS):
            raise ValueError(f"patient_history must contain one of the time columns: {TIME_COLUMNS}")

        # Basic dtype validation for numeric columns where present
        for col in df.columns:
            if col in ("Patient_ID",) or df[col].dtype == object:
                continue

        return df

    def analyze(self, patient_history: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the provided patient history and return a structured risk estimate.

        The latest observation (chronologically) is used as the inference timepoint and only
        features available up to that row are used to generate temporal features.
        """
        df = self._validate_history(patient_history)
        patient_id = str(df[PATIENT_ID_COLUMN].dropna().unique()[0])

        # Generate temporal features using the project's canonical functions
        try:
            temporal_df = add_temporal_features(df.copy())
        except Exception as exc:  # pragma: no cover - bubble errors to caller for clarity
            raise ValueError(f"Failed to generate temporal features: {exc}")

        # Sort and select the latest observation for this patient
        temporal_df = sort_patient_observations(temporal_df)
        latest_row = temporal_df[temporal_df[PATIENT_ID_COLUMN] == temporal_df[PATIENT_ID_COLUMN].iloc[0]].iloc[-1:]

        # Build the feature vector in the same order used during training
        missing_features = [c for c in self.feature_columns if c not in latest_row.columns]
        if len(missing_features) == len(self.feature_columns):
            raise ValueError("No required features are present after temporal feature generation")

        # Create X with exactly the feature columns (missing columns will be present as NaN)
        X = pd.DataFrame({col: (latest_row[col].values if col in latest_row.columns else [np.nan]) for col in self.feature_columns})

        # Preprocess and predict
        X_processed = self.preprocessor.transform(X)
        probs = self.model.predict_proba(X_processed)[:, 1]
        prob = float(probs[0])
        if not (0.0 <= prob <= 1.0):
            raise ValueError("Model returned invalid probability outside [0,1]")

        prediction = int(prob >= self.threshold)
        risk_level = self.risk_map.get(prediction, "UNKNOWN")

        return {
            "agent": self.__class__.__name__,
            "patient_id": patient_id,
            "disease": "Sepsis",
            "prediction": prediction,
            "probability": prob,
            "threshold": float(self.threshold),
            "risk_level": risk_level,
            "status": "success",
        }
