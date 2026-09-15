from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import Booster, DMatrix

from src.agents.result_schema import PatientAssessmentResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADVANCED_MODEL_DIR = PROJECT_ROOT / "models" / "aki_xgboost"
ADVANCED_MODEL_PATH = ADVANCED_MODEL_DIR / "xgboost_model.json"
ADVANCED_PREPROCESSOR_PATH = ADVANCED_MODEL_DIR / "preprocessor.joblib"
ADVANCED_METADATA_PATH = ADVANCED_MODEL_DIR / "metadata.json"
BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "aki" / "aki_pipeline.pkl"
BASELINE_METADATA_PATH = PROJECT_ROOT / "models" / "aki" / "aki_metadata.json"


class AKIDetectionAgent:
    """Prototype AKI detection agent using XGBoost by default."""

    def __init__(self, model_path: str | Path | None = None, threshold: float | None = None, model: str = "advanced"):
        if model not in {"advanced", "baseline"}:
            raise ValueError("model must be 'advanced' or 'baseline'")
        self.model_name = model
        if model == "advanced":
            self.model_path = Path(model_path) if model_path else ADVANCED_MODEL_PATH
            self.metadata_path = ADVANCED_METADATA_PATH
            self._load_advanced_artifacts()
        else:
            self.model_path = Path(model_path) if model_path else BASELINE_MODEL_PATH
            self.metadata_path = BASELINE_METADATA_PATH
            self._load_baseline_artifacts()
        saved_threshold = self.threshold
        self.threshold = float(saved_threshold if threshold is None else threshold)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.risk_map = {1: "ELEVATED", 0: "LOWER"}

    def _read_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"AKI metadata file not found: {self.metadata_path}")
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def _load_advanced_artifacts(self) -> None:
        if not self.model_path.exists() or not ADVANCED_PREPROCESSOR_PATH.exists():
            raise FileNotFoundError("AKI XGBoost model or preprocessing artifact is missing")
        self.metadata = self._read_metadata()
        self.feature_names = list(self.metadata["feature_columns"])
        self.excluded_columns = set(self.metadata["excluded_columns"])
        self.required_target = self.metadata.get("target", "AKI_TARGET")
        self.threshold = float(self.metadata["selected_threshold"])
        self.preprocessor = joblib.load(ADVANCED_PREPROCESSOR_PATH)
        self.model = Booster()
        self.model.load_model(str(self.model_path))
        self.pipeline = None

    def _load_baseline_artifacts(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"AKI model file not found: {self.model_path}")
        try:
            self.pipeline = joblib.load(self.model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load AKI pipeline: {exc}") from exc
        if not hasattr(self.pipeline, "predict_proba"):
            raise ValueError("Loaded AKI pipeline does not support predict_proba()")
        self.metadata = self._read_metadata()
        self.feature_names = list(self.metadata.get("feature_names", []))
        self.excluded_columns = set(self.metadata.get("excluded_columns", []))
        self.required_target = self.metadata.get("target", "AKI_TARGET")
        self.threshold = 0.5

    def _error(self, message: str, patient_reference: str | None = None) -> dict[str, Any]:
        result = {"agent": self.__class__.__name__, "disease": "AKI", "status": "error", "message": message}
        if patient_reference:
            result["patient_reference"] = patient_reference
        return result

    def _validate_patient_data(self, patient_data: dict[str, Any]) -> tuple[str, pd.DataFrame] | dict[str, Any]:
        if patient_data is None or not isinstance(patient_data, dict):
            return self._error("patient_data must be a non-empty dictionary")
        patient_reference = patient_data.get("patient_reference")
        if not patient_reference or not isinstance(patient_reference, str):
            return self._error("patient_reference is required and must be a non-empty string")
        forbidden_fields = {self.required_target, *self.excluded_columns}
        supplied_forbidden = sorted(forbidden_fields.intersection(patient_data))
        if supplied_forbidden:
            return self._error(f"Forbidden fields supplied: {supplied_forbidden}", patient_reference)
        missing_features = [field for field in self.feature_names if field not in patient_data]
        if missing_features:
            return self._error(f"Missing required model features: {missing_features}", patient_reference)
        feature_values: dict[str, float] = {}
        for feature in self.feature_names:
            value = patient_data[feature]
            if value is None or isinstance(value, bool):
                return self._error(f"Feature {feature} must be numeric", patient_reference)
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return self._error(f"Feature {feature} must be numeric", patient_reference)
            if not np.isfinite(numeric_value):
                return self._error(f"Feature {feature} must be finite", patient_reference)
            feature_values[feature] = numeric_value
        return patient_reference, pd.DataFrame([feature_values], columns=self.feature_names)

    def analyze(self, patient_data: dict[str, Any]) -> dict[str, Any]:
        validated = self._validate_patient_data(patient_data)
        if isinstance(validated, dict):
            return validated
        patient_reference, features = validated
        try:
            if self.model_name == "advanced":
                transformed = self.preprocessor.transform(features)
                probability = float(self.model.predict(DMatrix(transformed))[0])
            else:
                probability = float(self.pipeline.predict_proba(features)[0, 1])
        except Exception as exc:
            return self._error(f"Failed to generate AKI probability: {exc}", patient_reference)
        if not 0.0 <= probability <= 1.0:
            return self._error("Model returned an invalid probability value", patient_reference)
        prediction = int(probability >= self.threshold)
        result = PatientAssessmentResult(
            patient_id=patient_reference,
            disease="AKI",
            model="XGBoost" if self.model_name == "advanced" else "Baseline",
            prediction=prediction,
            probability=probability,
            threshold=self.threshold,
            risk_level=self.risk_map[prediction],
            observation_count=1,
            assessment_time=None,
            status="success",
            message="Academic prototype output; not a medical diagnosis.",
        ).to_dict()
        return result | {"agent": self.__class__.__name__, "patient_reference": patient_reference}
