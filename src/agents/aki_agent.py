from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class AKIDetectionAgent:
    """Prototype AKI detection agent for cross-sectional AKI risk estimation."""

    def __init__(self, model_path: str | Path = "models/aki/aki_pipeline.pkl", threshold: float = 0.5):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"AKI model file not found: {self.model_path}")

        try:
            self.pipeline = joblib.load(self.model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load AKI pipeline: {exc}") from exc

        if not hasattr(self.pipeline, "predict_proba"):
            raise ValueError("Loaded AKI pipeline does not support predict_proba()")

        self.threshold = float(threshold)
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be between 0 and 1")

        self.metadata_path = Path(__file__).resolve().parents[2] / "models" / "aki" / "aki_metadata.json"
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"AKI metadata file not found: {self.metadata_path}")

        with self.metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.feature_names = list(metadata.get("feature_names", []))
        self.excluded_columns = set(metadata.get("excluded_columns", []))
        self.required_target = metadata.get("target", "AKI_TARGET")
        self.target_mapping = metadata.get("target_mapping", {})

        if not self.feature_names:
            raise ValueError("AKI metadata does not contain feature_names")

        if any(col in self.excluded_columns for col in self.feature_names):
            raise ValueError("AKI feature_names include excluded target-defining variables")

        self.risk_map = {1: "ELEVATED", 0: "LOWER"}

    def _validate_patient_data(self, patient_data: dict[str, Any]) -> tuple[str, pd.DataFrame] | dict[str, Any]:
        if patient_data is None or not isinstance(patient_data, dict):
            return {
                "agent": self.__class__.__name__,
                "disease": "AKI",
                "status": "error",
                "message": "patient_data must be a non-empty dictionary",
            }

        patient_reference = patient_data.get("patient_reference")
        if not patient_reference or not isinstance(patient_reference, str):
            return {
                "agent": self.__class__.__name__,
                "disease": "AKI",
                "status": "error",
                "message": "patient_reference is required and must be a non-empty string",
            }

        forbidden_fields = [self.required_target] + list(self.excluded_columns)
        supplied_forbidden = [field for field in forbidden_fields if field in patient_data]
        if supplied_forbidden:
            return {
                "agent": self.__class__.__name__,
                "patient_reference": patient_reference,
                "disease": "AKI",
                "status": "error",
                "message": f"Forbidden fields supplied: {supplied_forbidden}",
            }

        missing_features = [field for field in self.feature_names if field not in patient_data]
        if missing_features:
            return {
                "agent": self.__class__.__name__,
                "patient_reference": patient_reference,
                "disease": "AKI",
                "status": "error",
                "message": f"Missing required model features: {missing_features}",
            }

        feature_values: dict[str, Any] = {}
        for feature in self.feature_names:
            value = patient_data[feature]
            if value is None or isinstance(value, bool):
                return {
                    "agent": self.__class__.__name__,
                    "patient_reference": patient_reference,
                    "disease": "AKI",
                    "status": "error",
                    "message": f"Feature {feature} must be a numeric value",
                }
            if isinstance(value, (int, float)):
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    return {
                        "agent": self.__class__.__name__,
                        "patient_reference": patient_reference,
                        "disease": "AKI",
                        "status": "error",
                        "message": f"Feature {feature} must be a finite numeric value",
                    }
                feature_values[feature] = float(value)
            else:
                try:
                    feature_values[feature] = float(value)
                except (TypeError, ValueError):
                    return {
                        "agent": self.__class__.__name__,
                        "patient_reference": patient_reference,
                        "disease": "AKI",
                        "status": "error",
                        "message": f"Feature {feature} must be numeric",
                    }

        X = pd.DataFrame([feature_values], columns=self.feature_names)
        return patient_reference, X

    def analyze(self, patient_data: dict[str, Any]) -> dict[str, Any]:
        validated = self._validate_patient_data(patient_data)
        if isinstance(validated, dict) and validated.get("status") == "error":
            return validated

        patient_reference, X = validated

        try:
            probabilities = self.pipeline.predict_proba(X)
        except Exception as exc:
            return {
                "agent": self.__class__.__name__,
                "patient_reference": patient_reference,
                "disease": "AKI",
                "status": "error",
                "message": f"Failed to generate AKI probability: {exc}",
            }

        if probabilities.shape[1] < 2:
            return {
                "agent": self.__class__.__name__,
                "patient_reference": patient_reference,
                "disease": "AKI",
                "status": "error",
                "message": "AKI pipeline did not return class probabilities for the positive class",
            }

        probability = float(probabilities[0, 1])
        if not (0.0 <= probability <= 1.0):
            return {
                "agent": self.__class__.__name__,
                "patient_reference": patient_reference,
                "disease": "AKI",
                "status": "error",
                "message": "Model returned an invalid probability value",
            }

        prediction = int(probability >= self.threshold)
        risk_level = self.risk_map[prediction]

        return {
            "agent": self.__class__.__name__,
            "patient_reference": patient_reference,
            "disease": "AKI",
            "prediction": prediction,
            "probability": probability,
            "threshold": self.threshold,
            "risk_level": risk_level,
            "status": "success",
        }
