from __future__ import annotations

from pathlib import Path

from .config import (
    BINARY_TARGET_COLUMN,
    POTENTIAL_TARGET_DEFINING_COLUMNS,
    RAW_AKI_DATA_PATH,
    TEST_SIZE,
)
from .preprocessing import create_binary_target, get_strict_feature_columns, load_raw_dataset
from .split_data import prepare_train_test_split


def test_train_test_counts_sum_to_total() -> None:
    df = load_raw_dataset()
    X_train, X_test, y_train, y_test, summary = prepare_train_test_split()
    assert summary["train_rows"] + summary["test_rows"] == df.shape[0]
    assert X_train.shape[0] == summary["train_rows"]
    assert X_test.shape[0] == summary["test_rows"]


def test_no_train_test_overlap() -> None:
    X_train, X_test, _, _, _ = prepare_train_test_split()
    assert X_train.index.intersection(X_test.index).empty


def test_both_classes_present_in_train_and_test() -> None:
    _, _, y_train, y_test, _ = prepare_train_test_split()
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_stratified_prevalence_similarity() -> None:
    _, _, y_train, y_test, summary = prepare_train_test_split()
    train_prev = summary["train_aki_prevalence"]
    test_prev = summary["test_aki_prevalence"]
    assert abs(train_prev - test_prev) < 0.02, "Train/test prevalence should be approximately similar"


def test_random_state_reproducibility() -> None:
    _, _, y_train_1, y_test_1, _ = prepare_train_test_split()
    _, _, y_train_2, y_test_2, _ = prepare_train_test_split()
    assert y_train_1.reset_index(drop=True).equals(y_train_2.reset_index(drop=True))
    assert y_test_1.reset_index(drop=True).equals(y_test_2.reset_index(drop=True))


def test_no_patient_id_required() -> None:
    df = load_raw_dataset()
    assert "Patient_ID" not in df.columns


def test_target_not_in_features() -> None:
    df = load_raw_dataset()
    strict_features = get_strict_feature_columns(df)
    assert BINARY_TARGET_COLUMN not in strict_features


def test_excluded_leakage_columns_not_in_features() -> None:
    df = load_raw_dataset()
    strict_features = get_strict_feature_columns(df)
    for col in POTENTIAL_TARGET_DEFINING_COLUMNS + ["aki_stage"]:
        assert col not in strict_features


if __name__ == "__main__":
    test_train_test_counts_sum_to_total()
    test_no_train_test_overlap()
    test_both_classes_present_in_train_and_test()
    test_stratified_prevalence_similarity()
    test_random_state_reproducibility()
    test_no_patient_id_required()
    test_target_not_in_features()
    test_excluded_leakage_columns_not_in_features()
    print("AKI split tests passed")
