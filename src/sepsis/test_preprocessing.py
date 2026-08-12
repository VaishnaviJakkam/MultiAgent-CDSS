import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sepsis.config import FEATURE_COLUMNS, PATIENT_ID_COLUMN, TARGET, TIME_COLUMNS
from src.sepsis.preprocessing import (
    build_preprocessing_pipeline,
    fit_preprocessor,
    load_dataset,
    split_features_target,
    transform_features,
)


def test_preprocessing_workflow() -> None:
    df = load_dataset()

    assert TARGET in df.columns, f"Expected target column {TARGET} in dataset"
    assert PATIENT_ID_COLUMN in df.columns, f"Expected patient ID column {PATIENT_ID_COLUMN} in dataset"
    assert all(col in df.columns for col in FEATURE_COLUMNS), "Some feature columns are missing from Dataset.csv"
    assert all(col in df.columns for col in TIME_COLUMNS), "Expected time context columns are missing"

    features, target, patient_ids, time_context = split_features_target(df)
    assert features.shape[0] == df.shape[0], "Feature count does not match input rows"
    assert target.shape[0] == df.shape[0], "Target count does not match input rows"
    assert patient_ids.shape[0] == df.shape[0], "Patient ID count does not match input rows"
    assert time_context.shape[0] == df.shape[0], "Time context count does not match input rows"

    assert PATIENT_ID_COLUMN not in features.columns, "Patient_ID must not be included in feature matrix"
    assert TARGET not in features.columns, "Target must not be included in feature matrix"

    preprocessor = build_preprocessing_pipeline()
    fitted = fit_preprocessor(preprocessor, features)
    transformed = transform_features(fitted, features)
    assert isinstance(transformed, np.ndarray), "Transformed output must be a NumPy array"
    assert transformed.shape[0] == features.shape[0], "Transformed row count mismatch"

    expected_feature_count = len(FEATURE_COLUMNS)
    if hasattr(fitted.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"], "get_feature_names_out"):
        expected_feature_count = len(features.columns.drop("Gender")) + 1
    assert transformed.shape[1] == expected_feature_count, "Transformed feature count mismatch"

    print("Preprocessing test passed")


if __name__ == "__main__":
    test_preprocessing_workflow()
