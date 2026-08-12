import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.sepsis.config import DEFAULT_CLASSIFICATION_THRESHOLD, FEATURE_COLUMNS, RANDOM_STATE, TARGET
from src.sepsis.predict import predict_sepsis_risk
from src.sepsis.split_data import load_raw_dataset, patient_level_train_test_split


def _build_sample_record() -> dict:
    df = load_raw_dataset()
    patient = df.drop_duplicates(subset=["Patient_ID"]).iloc[0].to_dict()
    record = {col: patient.get(col, 0.0) for col in FEATURE_COLUMNS}
    return record


def test_threshold_workflow() -> None:
    df = load_raw_dataset()
    X_train, X_test, y_train, y_test, train_ids, test_ids = patient_level_train_test_split(df)

    assert len(train_ids) > 0
    assert len(test_ids) > 0
    assert set(train_ids) & set(test_ids) == set()

    threshold = DEFAULT_CLASSIFICATION_THRESHOLD
    assert 0.0 < threshold < 1.0

    sample = _build_sample_record()
    result = predict_sepsis_risk(sample)
    assert set(result.keys()) >= {"prediction", "probability", "threshold", "risk_level", "target"}
    assert result["threshold"] == DEFAULT_CLASSIFICATION_THRESHOLD
    assert 0.0 <= result["probability"] <= 1.0
    assert result["prediction"] in {0, 1}
    assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}

    with open(Path("models") / "sepsis_metadata.json", "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    assert metadata.get("selected_threshold", 0.0) >= 0.0
    assert metadata.get("selected_threshold", 0.0) <= 1.0

    invalid = sample.copy()
    invalid.pop("HR")
    try:
        predict_sepsis_risk(invalid)
        raise AssertionError("Expected missing-feature validation error")
    except ValueError:
        pass

    print("Threshold validation test passed")


if __name__ == "__main__":
    test_threshold_workflow()
