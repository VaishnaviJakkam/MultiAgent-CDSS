from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    PATIENT_ID_COLUMN,
    TARGET,
    TIME_COLUMNS,
)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw sepsis dataset from the configured path."""
    if not path.exists():
        raise FileNotFoundError(f"Sepsis dataset not found at {path}")
    df = pd.read_csv(path, low_memory=False)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Separate features, target, patient IDs, and temporal/context fields."""
    missing_columns = [col for col in FEATURE_COLUMNS + [TARGET, PATIENT_ID_COLUMN] + TIME_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    features = df[FEATURE_COLUMNS].copy()
    target = df[TARGET].copy()
    patient_ids = df[PATIENT_ID_COLUMN].copy()
    time_context = df[TIME_COLUMNS].copy()
    return features, target, patient_ids, time_context


def build_preprocessing_pipeline() -> Pipeline:
    """Build a reusable sklearn preprocessing pipeline for sepsis features."""
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    return Pipeline([("preprocessor", preprocessor)])


def fit_preprocessor(preprocessor: Pipeline, features: pd.DataFrame) -> Pipeline:
    """Fit the preprocessing pipeline to feature data."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    return preprocessor.fit(features)


def transform_features(preprocessor: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Apply the fitted preprocessing pipeline to feature data."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    transformed = preprocessor.transform(features)
    if isinstance(transformed, np.matrix):
        transformed = np.asarray(transformed)
    return np.asarray(transformed)


def load_and_prepare_data() -> tuple[np.ndarray, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, Pipeline]:
    """Load the raw dataset, split it, and fit the preprocessing pipeline."""
    df = load_dataset()
    features, target, patient_ids, time_context = split_features_target(df)
    preprocessor = build_preprocessing_pipeline()
    fitted_preprocessor = fit_preprocessor(preprocessor, features)
    transformed_features = transform_features(fitted_preprocessor, features)
    return transformed_features, target, patient_ids, time_context, features, fitted_preprocessor


def get_feature_names(preprocessor: Pipeline) -> list[str]:
    """Return the transformed feature names after preprocessing."""
    transformer = preprocessor.named_steps["preprocessor"]
    feature_names: list[str] = []

    if "num" in transformer.named_transformers_:
        feature_names.extend(NUMERICAL_FEATURES)

    if "cat" in transformer.named_transformers_:
        cat_transformer = transformer.named_transformers_["cat"].named_steps["onehot"]
        encoded_cols = cat_transformer.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        feature_names.extend(encoded_cols)

    return feature_names
