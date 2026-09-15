from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.sepsis.config import DEFAULT_CLASSIFICATION_THRESHOLD, PATIENT_ID_COLUMN, TIME_COLUMNS
from src.sepsis.temporal_features import add_temporal_features, sort_patient_observations
from src.agents.result_schema import PatientAssessmentResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LSTM_MODEL_DIR = PROJECT_ROOT / "models" / "sepsis_lstm"
LSTM_MODEL_PATH = LSTM_MODEL_DIR / "sepsis_lstm.keras"
LSTM_PREPROCESSOR_PATH = LSTM_MODEL_DIR / "preprocessor.joblib"
LSTM_METADATA_PATH = LSTM_MODEL_DIR / "inference_metadata.joblib"
BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "temporal" / "sepsis_temporal_pipeline.pkl"
LSTM_FEATURES = ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp", "WBC", "Lactate"]
LSTM_SEQUENCE_LENGTH = 12


class SepsisDetectionAgent:
    """Prototype Sepsis detection agent using the LSTM by default."""

    def __init__(self, model_path: str | Path | None = None, threshold: float | None = None, risk_map: dict[int, str] | None = None, model: str = "advanced"):
        if model not in {"advanced", "baseline"}:
            raise ValueError("model must be 'advanced' or 'baseline'")
        self.model_name = model
        self.risk_map = risk_map or {1: "ELEVATED", 0: "LOWER"}
        self.model_path = Path(model_path) if model_path else (LSTM_MODEL_PATH if model == "advanced" else BASELINE_MODEL_PATH)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if model == "advanced":
            self._load_lstm_artifacts()
        else:
            self._load_baseline_artifacts()
        saved_threshold = self.threshold
        self.threshold = float(saved_threshold if threshold is None else threshold)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

    def _load_lstm_artifacts(self) -> None:
        if not LSTM_PREPROCESSOR_PATH.exists() or not LSTM_METADATA_PATH.exists():
            raise FileNotFoundError("Sepsis LSTM preprocessing or inference metadata artifact is missing")
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load Sepsis LSTM model: {exc}") from exc
        self.preprocessor = joblib.load(LSTM_PREPROCESSOR_PATH)
        metadata = joblib.load(LSTM_METADATA_PATH)
        self.feature_columns = list(metadata.get("feature_columns", LSTM_FEATURES))
        self.sequence_length = int(metadata.get("sequence_length", LSTM_SEQUENCE_LENGTH))
        self.threshold = float(metadata.get("threshold", DEFAULT_CLASSIFICATION_THRESHOLD))
        if self.feature_columns != LSTM_FEATURES:
            raise ValueError("Saved LSTM feature contract does not match the agent feature contract")
        if self.sequence_length != LSTM_SEQUENCE_LENGTH:
            raise ValueError("Unsupported saved LSTM sequence length")

    def _load_baseline_artifacts(self) -> None:
        payload = joblib.load(self.model_path)
        if not isinstance(payload, dict) or not {"preprocessor", "model", "feature_columns"}.issubset(payload):
            raise ValueError("Saved baseline Sepsis artifact has an unexpected format")
        self.pipeline = payload
        self.preprocessor = payload["preprocessor"]
        self.model = payload["model"]
        self.feature_columns = list(payload["feature_columns"])
        self.threshold = float(payload.get("threshold", DEFAULT_CLASSIFICATION_THRESHOLD))

    def _validate_history(self, patient_history: Iterable[dict[str, Any]]) -> pd.DataFrame:
        if patient_history is None:
            raise ValueError("patient_history is required and cannot be None")
        df = pd.DataFrame(list(patient_history))
        if df.empty:
            raise ValueError("patient_history must contain at least one observation")
        if PATIENT_ID_COLUMN not in df.columns:
            raise ValueError(f"Missing required column: {PATIENT_ID_COLUMN}")
        patient_ids = df[PATIENT_ID_COLUMN].dropna().unique()
        if len(patient_ids) == 0:
            raise ValueError(f"No valid {PATIENT_ID_COLUMN} values found")
        if len(patient_ids) > 1:
            raise ValueError("Inconsistent Patient_ID values found in patient_history")
        if not any(column in df.columns for column in TIME_COLUMNS):
            raise ValueError(f"patient_history must contain one of the time columns: {TIME_COLUMNS}")
        if self.model_name == "advanced":
            missing = [column for column in self.feature_columns if column not in df.columns]
            if missing:
                raise ValueError(f"Missing required LSTM features: {missing}")
            if len(df) < self.sequence_length:
                raise ValueError(f"Sepsis LSTM requires at least {self.sequence_length} real observations; received {len(df)}")
            if "Hour" not in df.columns or df["Hour"].isna().any():
                raise ValueError("Hour values must be present for chronological LSTM inference")
        return df

    def _predict_lstm(self, df: pd.DataFrame) -> float:
        ordered = df.sort_values([PATIENT_ID_COLUMN, "Hour"], kind="mergesort").tail(self.sequence_length)
        transformed = np.asarray(self.preprocessor.transform(ordered[self.feature_columns]), dtype=np.float32)
        probabilities = self.model.predict(transformed[np.newaxis, :, :], verbose=0).reshape(-1)
        if probabilities.size != 1:
            raise ValueError("Sepsis LSTM returned an unexpected prediction shape")
        return float(probabilities[0])

    def _predict_baseline(self, df: pd.DataFrame) -> float:
        temporal_df = sort_patient_observations(add_temporal_features(df.copy()))
        latest_row = temporal_df.iloc[-1:]
        features = pd.DataFrame({column: latest_row[column].values if column in latest_row else [np.nan] for column in self.feature_columns})
        return float(self.model.predict_proba(self.preprocessor.transform(features))[:, 1][0])

    def analyze(self, patient_history: Iterable[dict[str, Any]]) -> dict[str, Any]:
        df = self._validate_history(patient_history)
        patient_id = str(df[PATIENT_ID_COLUMN].dropna().unique()[0])
        probability = self._predict_lstm(df) if self.model_name == "advanced" else self._predict_baseline(df)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Model returned invalid probability outside [0,1]")
        prediction = int(probability >= self.threshold)
        observation_count = self.sequence_length if self.model_name == "advanced" else len(df)
        ordered = df.sort_values([PATIENT_ID_COLUMN, "Hour"], kind="mergesort")
        assessment_time = ordered["Hour"].iloc[-1] if "Hour" in ordered else None
        if isinstance(assessment_time, np.generic):
            assessment_time = assessment_time.item()
        return PatientAssessmentResult(
            patient_id=patient_id,
            disease="Sepsis",
            model="LSTM" if self.model_name == "advanced" else "Temporal Logistic Regression",
            prediction=prediction,
            probability=probability,
            threshold=float(self.threshold),
            risk_level=self.risk_map.get(prediction, "UNKNOWN"),
            observation_count=observation_count,
            assessment_time=assessment_time,
            status="success",
            message="Academic prototype output; not a medical diagnosis.",
        ).to_dict() | {"agent": self.__class__.__name__, "number_of_observations_used": observation_count}
