from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    BINARY_TARGET_COLUMN,
    BINARY_TARGET_MAPPING,
    POTENTIAL_TARGET_DEFINING_COLUMNS,
    RAW_AKI_DATA_PATH,
    STRICT_BASELINE_EXCLUDED_COLUMNS,
    TARGET_COLUMN,
)
from .preprocessing import (
    build_preprocessing_pipeline,
    create_binary_target,
    get_feature_types,
    get_strict_feature_columns,
    load_raw_dataset,
    validate_raw_dataset,
)


def test_raw_dataset_loads_successfully() -> None:
    df = load_raw_dataset()
    assert df.shape[0] > 0, "Raw AKI dataset should contain rows"
    assert df.shape[1] > 0, "Raw AKI dataset should contain columns"
    assert RAW_AKI_DATA_PATH.exists(), "Raw AKI CSV path must exist"


def test_binary_target_generation() -> None:
    df = load_raw_dataset()
    validate_raw_dataset(df)
    binary_target = create_binary_target(df)
    assert binary_target.name == BINARY_TARGET_COLUMN
    assert binary_target.dtype == int
    assert binary_target.isna().sum() == 0, "Binary target must have no missing values"
    assert set(binary_target.unique()) <= {0, 1}
    counts = binary_target.value_counts().to_dict()
    assert counts[0] + counts[1] == df.shape[0]


def test_aki_stage_not_in_model_features() -> None:
    df = load_raw_dataset()
    strict_features = get_strict_feature_columns(df)
    assert TARGET_COLUMN not in strict_features, "aki_stage must not be present in strict baseline features"


def test_potential_target_defining_columns_are_excluded() -> None:
    df = load_raw_dataset()
    strict_features = get_strict_feature_columns(df)
    for col in POTENTIAL_TARGET_DEFINING_COLUMNS:
        assert col not in strict_features, f"{col} must be excluded from strict baseline features"


def test_no_patient_id_or_temporal_columns_required() -> None:
    df = load_raw_dataset()
    assert "Patient_ID" not in df.columns, "Dataset must not require Patient_ID"
    assert "Hour" not in df.columns or "ICULOS" not in df.columns, "Dataset must not require temporal order fields"


def test_numerical_feature_detection() -> None:
    df = load_raw_dataset()
    strict_features = get_strict_feature_columns(df)
    numerical, categorical = get_feature_types(df, strict_features)
    assert len(strict_features) == len(numerical) + len(categorical)
    assert all(df[col].dtype.kind in "biufc" for col in numerical)
    assert isinstance(categorical, list)


def test_missing_value_preprocessing_pipeline_configuration() -> None:
    df = load_raw_dataset()
    strict_features = get_strict_feature_columns(df)
    numerical, categorical = get_feature_types(df, strict_features)
    pipeline = build_preprocessing_pipeline(numerical, categorical)
    assert pipeline is not None
    assert hasattr(pipeline, "fit"), "Pipeline must support fit"
    assert hasattr(pipeline, "transform"), "Pipeline must support transform"


def test_raw_dataset_not_modified() -> None:
    path = RAW_AKI_DATA_PATH
    before = path.read_bytes()
    _ = load_raw_dataset()
    after = path.read_bytes()
    assert before == after, "Loading must not modify the raw CSV file"


if __name__ == "__main__":
    test_raw_dataset_loads_successfully()
    test_binary_target_generation()
    test_aki_stage_not_in_model_features()
    test_potential_target_defining_columns_are_excluded()
    test_no_patient_id_or_temporal_columns_required()
    test_numerical_feature_detection()
    test_missing_value_preprocessing_pipeline_configuration()
    test_raw_dataset_not_modified()
    print("AKI preprocessing tests passed")
