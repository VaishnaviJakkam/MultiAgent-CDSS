from __future__ import annotations

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TEST_SIZE
from .preprocessing import (
    BINARY_TARGET_COLUMN,
    create_binary_target,
    get_strict_feature_columns,
    load_raw_dataset,
)

SPLIT_SUMMARY_PATH = Path(__file__).resolve().parents[2] / "models" / "aki" / "aki_split_summary.json"


def prepare_train_test_split() -> tuple[Path, dict[str, int | float]]:
    """Prepare a stratified observation-level train/test split for the AKI dataset."""
    df = load_raw_dataset()
    y = create_binary_target(df)
    X = df[get_strict_feature_columns(df)].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    summary = {
        "total_rows": int(df.shape[0]),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "train_no_aki": int((y_train == 0).sum()),
        "train_aki": int((y_train == 1).sum()),
        "test_no_aki": int((y_test == 0).sum()),
        "test_aki": int((y_test == 1).sum()),
        "train_aki_prevalence": float((y_train == 1).mean()),
        "test_aki_prevalence": float((y_test == 1).mean()),
        "test_size": float(TEST_SIZE),
        "random_state": int(RANDOM_STATE),
        "split_type": "stratified_observation_level",
    }

    save_split_summary(summary)
    return X_train, X_test, y_train, y_test, summary


def save_split_summary(summary: dict[str, int | float], path: Path = SPLIT_SUMMARY_PATH) -> None:
    """Save the split summary summary to a compact JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def verify_no_overlap(X_train: Path | object, X_test: Path | object) -> bool:
    """Verify that the train and test splits do not share row indices."""
    train_index = X_train.index
    test_index = X_test.index
    return train_index.intersection(test_index).empty


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, summary = prepare_train_test_split()
    overlap = X_train.index.intersection(X_test.index)
    print(f"Total rows: {summary['total_rows']}")
    print(f"Train rows: {summary['train_rows']}")
    print(f"Test rows: {summary['test_rows']}")
    print(f"Train No AKI: {summary['train_no_aki']}")
    print(f"Train AKI: {summary['train_aki']}")
    print(f"Train AKI prevalence: {summary['train_aki_prevalence']:.4f}")
    print(f"Test No AKI: {summary['test_no_aki']}")
    print(f"Test AKI: {summary['test_aki']}")
    print(f"Test AKI prevalence: {summary['test_aki_prevalence']:.4f}")
    print(f"Train/test overlap: {len(overlap)} rows")
    print("Split method: stratified observation-level train_test_split")
    print("The dataset contains no patient identifier, so the split is stratified at the observation level.")
    print(f"Split summary saved to {SPLIT_SUMMARY_PATH}")
