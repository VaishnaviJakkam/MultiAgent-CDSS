from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    BINARY_TARGET_COLUMN,
    BINARY_TARGET_MAPPING,
    CLINICAL_EXTENDED_EXCLUDED_COLUMNS,
    EXCLUDED_COLUMNS,
    POTENTIAL_TARGET_DEFINING_COLUMNS,
    RAW_AKI_DATA_PATH,
    STRICT_BASELINE_EXCLUDED_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
)


def load_raw_dataset(path: Path = RAW_AKI_DATA_PATH) -> pd.DataFrame:
    """Load the raw AKI dataset without modifying the source CSV."""
    if not path.exists():
        raise FileNotFoundError(f"AKI dataset not found at {path}")
    return pd.read_csv(path, low_memory=False)


def validate_raw_dataset(df: pd.DataFrame) -> None:
    """Validate that the raw dataset contains the expected AKI target column."""
    missing = [col for col in [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f"{TARGET_COLUMN} contains missing values")
    unexpected = sorted(set(df[TARGET_COLUMN].dropna().unique()) - set(BINARY_TARGET_MAPPING.keys()))
    if unexpected:
        raise ValueError(f"Unexpected {TARGET_COLUMN} values: {unexpected}")


def create_binary_target(df: pd.DataFrame) -> pd.Series:
    """Create the AKI binary detection target from the raw aki_stage column."""
    validate_raw_dataset(df)
    binary_target = df[TARGET_COLUMN].map(BINARY_TARGET_MAPPING)
    if binary_target.isna().any():
        raise ValueError("Binary target conversion produced missing values")
    return pd.Series(binary_target.astype(int), name=BINARY_TARGET_COLUMN, index=df.index)


def detect_invalid_feature_columns(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    """Detect clearly invalid feature columns that should be reviewed before modeling."""
    constant_columns = [col for col in feature_columns if df[col].nunique(dropna=False) <= 1]
    all_missing_columns = [col for col in feature_columns if df[col].isna().all()]
    duplicate_pairs: list[tuple[str, str]] = []
    for i, col1 in enumerate(feature_columns):
        for col2 in feature_columns[i + 1 :]:
            if df[col1].equals(df[col2]):
                duplicate_pairs.append((col1, col2))
    return {
        "constant_columns": constant_columns,
        "all_missing_columns": all_missing_columns,
        "duplicate_column_pairs": duplicate_pairs,
    }


def get_strict_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the strict baseline feature set excluding likely target-defining variables."""
    feature_columns = [col for col in df.columns if col not in STRICT_BASELINE_EXCLUDED_COLUMNS]
    invalid = detect_invalid_feature_columns(df, feature_columns)
    return [
        col
        for col in feature_columns
        if col not in invalid["constant_columns"] and col not in invalid["all_missing_columns"]
    ]


def get_extended_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the clinical extended feature set for later evaluation."""
    feature_columns = [col for col in df.columns if col not in CLINICAL_EXTENDED_EXCLUDED_COLUMNS]
    invalid = detect_invalid_feature_columns(df, feature_columns)
    return [
        col
        for col in feature_columns
        if col not in invalid["constant_columns"] and col not in invalid["all_missing_columns"]
    ]


def get_feature_types(df: pd.DataFrame, feature_columns: list[str]) -> tuple[list[str], list[str]]:
    """Separate feature columns into numerical and categorical sets based on actual data types."""
    numerical = [col for col in feature_columns if is_numeric_dtype(df[col].dtype)]
    categorical = [col for col in feature_columns if not is_numeric_dtype(df[col].dtype)]
    return numerical, categorical


def build_preprocessing_pipeline(numerical_features: list[str], categorical_features: list[str]) -> Pipeline:
    """Build a reusable preprocessing pipeline for the AKI feature set."""
    transformers = []
    if numerical_features:
        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_transformer, numerical_features))

    if categorical_features:
        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)
    return Pipeline([("preprocessor", preprocessor)])


def get_preprocessing_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """Return summary information for the current AKI preprocessing definition."""
    strict_features = get_strict_feature_columns(df)
    extended_features = get_extended_feature_columns(df)
    numerical, categorical = get_feature_types(df, strict_features)
    return {
        "raw_row_count": int(df.shape[0]),
        "raw_column_count": int(df.shape[1]),
        "binary_target_column": BINARY_TARGET_COLUMN,
        "strict_feature_count": len(strict_features),
        "extended_feature_count": len(extended_features),
        "numerical_feature_count": len(numerical),
        "categorical_feature_count": len(categorical),
        "strict_feature_columns": strict_features,
        "extended_feature_columns": extended_features,
        "excluded_columns": STRICT_BASELINE_EXCLUDED_COLUMNS,
        "missing_value_strategy": "SimpleImputer(strategy='median') for numerical; SimpleImputer(strategy='most_frequent') for categorical",
        "encoding_strategy": "OneHotEncoder(handle_unknown='ignore') for categorical only",
        "scaling_strategy": "StandardScaler for numerical features",
    }
