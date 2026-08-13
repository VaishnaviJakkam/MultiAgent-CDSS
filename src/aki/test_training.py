from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .config import (
    BINARY_TARGET_COLUMN,
    POTENTIAL_TARGET_DEFINING_COLUMNS,
    RAW_AKI_DATA_PATH,
    STRICT_BASELINE_EXCLUDED_COLUMNS,
    TARGET_COLUMN,
)
from .preprocessing import create_binary_target, get_strict_feature_columns, load_raw_dataset
from .split_data import prepare_train_test_split
from .train import (
    METADATA_JSON_PATH,
    METRICS_CSV_PATH,
    METRICS_JSON_PATH,
    PIPELINE_PATH,
    CONFUSION_MATRIX_FILES,
    run_baseline_training,
)

_cached_training_result: dict[str, object] | None = None


def baseline_training_result() -> dict[str, object]:
    global _cached_training_result
    if _cached_training_result is None:
        _cached_training_result = run_baseline_training()
    return _cached_training_result


def test_training_data_loads() -> None:
    df = load_raw_dataset()
    assert RAW_AKI_DATA_PATH.exists()
    assert df.shape[0] > 0
    assert df.shape[1] > 0


def test_target_is_binary() -> None:
    df = load_raw_dataset()
    target = create_binary_target(df)
    assert set(target.unique()) <= {0, 1}
    assert target.isna().sum() == 0


def test_no_target_leakage_in_features() -> None:
    X_train, X_test, _, _, _ = prepare_train_test_split()
    assert BINARY_TARGET_COLUMN not in X_train.columns
    assert BINARY_TARGET_COLUMN not in X_test.columns
    for excluded in STRICT_BASELINE_EXCLUDED_COLUMNS:
        assert excluded not in X_train.columns
        assert excluded not in X_test.columns


def test_excluded_leakage_columns_are_absent() -> None:
    df = load_raw_dataset()
    strict_features = get_strict_feature_columns(df)
    for excluded in POTENTIAL_TARGET_DEFINING_COLUMNS + [TARGET_COLUMN]:
        assert excluded not in strict_features


def test_model_training_completes_and_artifacts_exist() -> None:
    result = baseline_training_result()
    assert Path(result["pipeline_path"]).exists()
    assert Path(result["metrics_json_path"]).exists()
    assert Path(result["metadata_json_path"]).exists()
    assert Path(result["comparison_csv_path"]).exists()
    for path in result["confusion_matrix_paths"].values():
        assert Path(path).exists()


def test_probability_output_and_prediction_range() -> None:
    result = baseline_training_result()
    pipeline = joblib.load(result["pipeline_path"])
    X_train, X_test, _, _, _ = prepare_train_test_split()
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = pipeline.predict(X_test)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert set(np.unique(preds)).issubset({0, 1})


def test_metrics_and_metadata_content() -> None:
    result = baseline_training_result()
    metrics_path = Path(result["metrics_json_path"])
    metadata_path = Path(result["metadata_json_path"])
    assert metrics_path.exists()
    assert metadata_path.exists()
    metrics = metrics_path.read_text(encoding="utf-8")
    metadata = metadata_path.read_text(encoding="utf-8")
    assert "selected_model" in metrics
    assert "target" in metadata
    assert "excluded_columns" in metadata


if __name__ == "__main__":
    test_training_data_loads()
    test_target_is_binary()
    test_no_target_leakage_in_features()
    test_excluded_leakage_columns_are_absent()
    test_model_training_completes_and_artifacts_exist()
    test_probability_output_and_prediction_range()
    test_metrics_and_metadata_content()
    print("AKI training tests passed")
