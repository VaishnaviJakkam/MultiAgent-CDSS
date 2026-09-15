from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.aki.config import BINARY_TARGET_COLUMN, POTENTIAL_TARGET_DEFINING_COLUMNS, STRICT_BASELINE_EXCLUDED_COLUMNS
from src.aki.preprocessing import create_binary_target, get_strict_feature_columns, load_raw_dataset
from src.aki.split_data import prepare_train_test_split
from src.aki.xgboost_model import (
    XGBoostConfig,
    build_model,
    calculate_scale_pos_weight,
    evaluate_probabilities,
    fit_preprocessor,
    get_final_feature_columns,
    select_operating_threshold,
    split_training_validation,
    threshold_table,
    validate_feature_columns,
)


def test_final_features_exclude_target_and_leakage_columns() -> None:
    df = load_raw_dataset()
    features = get_final_feature_columns(df)
    assert BINARY_TARGET_COLUMN not in features
    assert set(features).isdisjoint(set(STRICT_BASELINE_EXCLUDED_COLUMNS))
    assert set(features).isdisjoint(set(POTENTIAL_TARGET_DEFINING_COLUMNS))
    validate_feature_columns(features)


def test_existing_stratified_split_has_no_row_overlap() -> None:
    X_train, X_test, y_train, y_test, summary = prepare_train_test_split()
    assert X_train.index.intersection(X_test.index).empty
    assert len(y_train) == summary["train_rows"]
    assert len(y_test) == summary["test_rows"]


def test_validation_split_is_stratified_and_separate() -> None:
    df = load_raw_dataset()
    target = create_binary_target(df)
    feature_frame = df[get_strict_feature_columns(df)]
    X_train, X_test, y_train, y_test, _ = prepare_train_test_split()
    train_part, validation_part, y_train_part, y_validation = split_training_validation(X_train, y_train)
    assert train_part.index.intersection(validation_part.index).empty
    assert train_part.index.intersection(X_test.index).empty
    assert validation_part.index.intersection(X_test.index).empty
    assert set(y_train_part.unique()) == {0, 1}
    assert set(y_validation.unique()) == {0, 1}
    assert len(target) == len(feature_frame)


def test_preprocessing_is_fitted_on_train_subset_only() -> None:
    X_train, X_test, _, _, _ = prepare_train_test_split()
    train_subset = X_train.iloc[:1000]
    preprocessor = fit_preprocessor(train_subset)
    transformed_test = preprocessor.transform(X_test.iloc[:5])
    assert transformed_test.shape[0] == 5
    assert transformed_test.shape[1] > 0


def test_model_output_shape_probability_range_and_artifact_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(80, 4))
    labels = np.array([0, 1] * 40)
    model = build_model(XGBoostConfig(n_estimators=5, early_stopping_rounds=2), calculate_scale_pos_weight(labels))
    model.fit(features[:60], labels[:60], eval_set=[(features[60:], labels[60:])], verbose=False)
    probabilities = model.predict_proba(features[60:])[:, 1]
    assert probabilities.shape == (20,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    path = tmp_path / "model.json"
    model.get_booster().save_model(path)
    loaded = XGBClassifier()
    loaded.load_model(path)
    assert loaded.predict_proba(features[60:]).shape == (20, 2)


def test_threshold_evaluation_and_selection() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.3, 0.7, 0.9])
    table = threshold_table(labels, probabilities, (0.2, 0.5, 0.8))
    assert list(table.columns) == ["threshold", "precision", "recall", "f1", "number_of_positive_predictions"]
    assert select_operating_threshold(table) in {0.2, 0.5, 0.8}
    metrics = evaluate_probabilities(labels, probabilities, 0.5)
    assert metrics["number_of_positive_predictions"] == 2
    assert metrics["recall"] == pytest.approx(1.0)