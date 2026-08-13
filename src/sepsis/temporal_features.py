from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import FEATURE_COLUMNS, PATIENT_ID_COLUMN, RANDOM_STATE, TARGET
from .split_data import load_raw_dataset, patient_level_train_test_split

TEMPORAL_BASE_COLUMNS = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "Resp",
    "WBC",
    "Lactate",
]

TEMPORAL_COLUMNS: list[str] = []


def sort_patient_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Sort each patient's rows chronologically without ever using future information.

    The ordering is time-based within each patient only. This is a causal step: later features are
    always computed from earlier rows of the same patient, never from future rows.
    """
    if PATIENT_ID_COLUMN not in df.columns:
        raise ValueError(f"Missing required patient column: {PATIENT_ID_COLUMN}")
    if "Hour" in df.columns:
        time_col = "Hour"
    elif "ICULOS" in df.columns:
        time_col = "ICULOS"
    else:
        raise ValueError("Dataset must contain either 'Hour' or 'ICULOS' for chronological ordering")

    out = df.copy()
    # Preserve the original patient order while sorting each patient's rows chronologically.
    patient_order = pd.Categorical(out[PATIENT_ID_COLUMN], categories=out[PATIENT_ID_COLUMN].drop_duplicates().tolist(), ordered=True)
    out = out.assign(_patient_order=patient_order)
    sort_columns = ["_patient_order", time_col]
    if "Unnamed: 0" in out.columns:
        sort_columns.append("Unnamed: 0")
    return out.sort_values(sort_columns, kind="mergesort").drop(columns=["_patient_order"]).reset_index(drop=True)


def _safe_pct_change(current: float, previous: float) -> float:
    if pd.isna(current) or pd.isna(previous):
        return np.nan
    if abs(previous) < 1e-8:
        return np.nan
    return float((current - previous) / abs(previous))


def add_temporal_features(df: pd.DataFrame, selected_columns: list[str] | None = None) -> pd.DataFrame:
    """Add causal lag, change, percentage-change, and rolling features for historical signals.

    All features are built using data available at the same row or earlier within the same patient.
    This function never uses future observations, future labels, or patient-level leakage.
    """
    out = sort_patient_observations(df).copy()
    selected = [col for col in (selected_columns or TEMPORAL_BASE_COLUMNS) if col in out.columns]
    if not selected:
        raise ValueError("No temporal columns are available in the dataframe")

    for col in selected:
        previous_name = f"previous_{col}"
        change_name = f"{col}_change"
        pct_name = f"{col}_pct_change"
        rolling_mean_name = f"{col}_rolling_mean_3"
        rolling_std_name = f"{col}_rolling_std_3"

        out[previous_name] = out.groupby(PATIENT_ID_COLUMN)[col].transform(lambda s: s.shift(1))
        out[change_name] = out.groupby(PATIENT_ID_COLUMN)[col].transform(lambda s: s - s.shift(1))

        def _pct_transform(s: pd.Series) -> pd.Series:
            prev = s.shift(1)
            safe = np.where(np.abs(prev) > 1e-8, (s - prev) / np.abs(prev), np.nan)
            return pd.Series(safe, index=s.index, dtype=float)

        out[pct_name] = out.groupby(PATIENT_ID_COLUMN)[col].transform(_pct_transform)
        out[rolling_mean_name] = out.groupby(PATIENT_ID_COLUMN)[col].transform(lambda s: s.rolling(window=3, min_periods=1).mean())
        out[rolling_std_name] = out.groupby(PATIENT_ID_COLUMN)[col].transform(lambda s: s.rolling(window=3, min_periods=1).std(ddof=0))

    generated = [
        col
        for col in out.columns
        if col.startswith(("previous_", "rolling_"))
        or col.endswith(("_change", "_pct_change", "_rolling_mean_3", "_rolling_std_3"))
    ]
    global TEMPORAL_COLUMNS
    TEMPORAL_COLUMNS.clear()
    TEMPORAL_COLUMNS.extend(generated)
    return out


def build_temporal_training_frame(df: pd.DataFrame, patient_ids: pd.Index | list[int] | set[int], selected_columns: list[str] | None = None) -> pd.DataFrame:
    """Create the feature matrix for a train/test patient subset without leaking patient identity or future labels."""
    patient_ids = pd.Index(sorted(set(patient_ids)))
    patient_mask = df[PATIENT_ID_COLUMN].isin(patient_ids)
    patient_df = df.loc[patient_mask].copy()
    temporal_df = add_temporal_features(patient_df, selected_columns=selected_columns)

    feature_candidates = [
        col
        for col in temporal_df.columns
        if col in FEATURE_COLUMNS
        or any(
            col.startswith(prefix)
            for prefix in ["previous_", "rolling_", "HR_", "O2Sat_", "Temp_", "SBP_", "MAP_", "DBP_", "Resp_", "Creatinine_", "BUN_", "WBC_", "Lactate_", "Platelets_", "Glucose_", "pH_", "BaseExcess_"]
        )
        or col.endswith(("_change", "_pct_change", "_rolling_mean_3", "_rolling_std_3"))
    ]
    feature_candidates = [col for col in feature_candidates if col not in {PATIENT_ID_COLUMN, TARGET, "Unnamed: 0", "Hour", "HospAdmTime", "ICULOS", "Unit1", "Unit2"}]
    return temporal_df[feature_candidates].copy()


def summarize_temporal_features(df: pd.DataFrame) -> dict[str, Any]:
    """Report how many causal temporal features were generated and the rate of missingness."""
    temporal_columns = [
        col
        for col in df.columns
        if col.startswith(("previous_", "rolling_"))
        or col.endswith(("_change", "_pct_change", "_rolling_mean_3", "_rolling_std_3"))
    ]
    if not temporal_columns:
        raise ValueError("No temporal features were generated")

    missing_fraction = float(df[temporal_columns].isna().mean().mean())
    return {
        "selected_columns": temporal_columns,
        "temporal_feature_count": len(temporal_columns),
        "missing_fraction": missing_fraction,
        "missing_by_feature": {col: float(df[col].isna().mean()) for col in temporal_columns[:10]},
    }


def export_temporal_summary(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a lightweight JSON summary of the temporal feature experiment."""
    summary = summarize_temporal_features(df)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_temporal_preprocessing_pipeline(feature_columns: list[str]) -> Pipeline:
    """Build a preprocessing pipeline for temporally expanded features while preserving train-only fitting."""
    numeric_columns = [col for col in feature_columns if col != "Gender"]
    categorical_columns = ["Gender"] if "Gender" in feature_columns else []

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore")),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = [("num", numeric_transformer, numeric_columns)]
    if categorical_columns:
        transformers.append(("cat", categorical_transformer, categorical_columns))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)
    return Pipeline([("preprocessor", preprocessor)])


def prepare_temporal_dataset(df: pd.DataFrame, patient_ids: pd.Index | list[int] | set[int]) -> tuple[pd.DataFrame, pd.Series, dict[str, Any], list[str]]:
    """Generate causal temporal features for a patient subset and return aligned X/y arrays."""
    patient_ids = pd.Index(sorted(set(patient_ids)))
    patient_subset = df[df[PATIENT_ID_COLUMN].isin(patient_ids)].copy()
    patient_subset = sort_patient_observations(patient_subset).reset_index(drop=True)
    temporal_df = add_temporal_features(patient_subset)

    feature_columns = [
        col
        for col in temporal_df.columns
        if col in FEATURE_COLUMNS
        or col.startswith(("previous_", "rolling_"))
        or col.endswith(("_change", "_pct_change", "_rolling_mean_3", "_rolling_std_3"))
    ]
    feature_columns = [col for col in feature_columns if col not in {PATIENT_ID_COLUMN, TARGET, "Unnamed: 0", "Hour", "HospAdmTime", "ICULOS", "Unit1", "Unit2"}]
    feature_columns = list(dict.fromkeys(feature_columns))

    X = temporal_df[feature_columns].copy()
    y = temporal_df[TARGET].copy()
    summary = summarize_temporal_features(temporal_df)
    return X, y, summary, feature_columns


def evaluate_model_metrics(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float = 0.50) -> dict[str, Any]:
    """Generate standard binary classification metrics at a fixed threshold."""
    y_true_array = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_array, preds, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_array, preds)),
        "precision": float(precision_score(y_true_array, preds, zero_division=0)),
        "recall": float(recall_score(y_true_array, preds, zero_division=0)),
        "f1": float(f1_score(y_true_array, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_array, probs) if len(np.unique(y_true_array)) > 1 else float("nan")),
        "pr_auc": float(average_precision_score(y_true_array, probs) if len(np.unique(y_true_array)) > 1 else float("nan")),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_temporal_experiment(output_dir: str | Path = "models/temporal", sample_fraction: float = 0.10) -> dict[str, Any]:
    """Train and evaluate a temporal-history logistic regression model in a separate experiment.

    The patient-level split is preserved exactly: all rows come from either the original train or
    test patients, never both. To keep the longitudinal experiment computationally manageable,
    rows are randomly sampled within each split after patient-level separation; this does not create
    train/test leakage and keeps the experiment reproducible.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_raw_dataset()
    X_train, X_test, y_train, y_test, train_ids, test_ids = patient_level_train_test_split(df)

    train_df = df[df[PATIENT_ID_COLUMN].isin(train_ids)].copy()
    test_df = df[df[PATIENT_ID_COLUMN].isin(test_ids)].copy()
    train_df = sort_patient_observations(train_df).reset_index(drop=True)
    test_df = sort_patient_observations(test_df).reset_index(drop=True)

    train_temporal = add_temporal_features(train_df)
    test_temporal = add_temporal_features(test_df)

    if 0.0 < sample_fraction < 1.0:
        train_temporal = train_temporal.sample(frac=sample_fraction, random_state=RANDOM_STATE).reset_index(drop=True)
        test_temporal = test_temporal.sample(frac=sample_fraction, random_state=RANDOM_STATE).reset_index(drop=True)

    feature_columns = [
        col
        for col in train_temporal.columns
        if col in FEATURE_COLUMNS or col.startswith(("previous_", "rolling_")) or col.endswith(("_change", "_pct_change", "_rolling_mean_3", "_rolling_std_3"))
    ]
    feature_columns = [col for col in feature_columns if col not in {PATIENT_ID_COLUMN, TARGET, "Unnamed: 0", "Hour", "HospAdmTime", "ICULOS", "Unit1", "Unit2"}]
    feature_columns = list(dict.fromkeys(feature_columns))

    X_train = train_temporal[feature_columns].copy()
    X_test = test_temporal[feature_columns].copy()
    y_train = train_temporal[TARGET].copy()
    y_test = test_temporal[TARGET].copy()

    temporal_summary = summarize_temporal_features(train_temporal)
    preprocessor = build_temporal_preprocessing_pipeline(feature_columns)
    fitted_preprocessor = preprocessor.fit(X_train)
    X_train_processed = fitted_preprocessor.transform(X_train)
    X_test_processed = fitted_preprocessor.transform(X_test)

    model = LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=200, random_state=RANDOM_STATE)
    model.fit(X_train_processed, y_train)

    test_proba = model.predict_proba(X_test_processed)[:, 1]
    temporal_metrics = evaluate_model_metrics(y_test, test_proba, threshold=0.50)

    payload = {
        "preprocessor": fitted_preprocessor,
        "model": model,
        "model_name": "LogisticRegression",
        "feature_columns": feature_columns,
        "target": TARGET,
        "threshold": 0.50,
    }
    pipeline_path = output_path / "sepsis_temporal_pipeline.pkl"
    joblib.dump(payload, pipeline_path)

    temporal_csv_path = output_path / "temporal_model_comparison.csv"
    baseline_csv = Path("models/sepsis_model_comparison.csv")
    baseline_df = pd.read_csv(baseline_csv)
    baseline_row = baseline_df[baseline_df["model"] == "LogisticRegression"].iloc[0].to_dict()
    temporal_row = {"model": "LogisticRegression (temporal)", **temporal_metrics}
    comparison_df = pd.DataFrame([baseline_row, temporal_row])
    comparison_df.to_csv(temporal_csv_path, index=False)

    temporal_json_path = output_path / "temporal_metrics.json"
    temporal_json_path.write_text(json.dumps({
        "baseline_reference": baseline_row,
        "temporal_model": temporal_row,
        "feature_summary": temporal_summary,
        "selected_features": feature_columns,
        "train_patients": len(train_ids),
        "test_patients": len(test_ids),
        "train_rows": len(train_temporal),
        "test_rows": len(test_temporal),
        "threshold": 0.50,
    }, indent=2), encoding="utf-8")

    summary_path = output_path / "temporal_feature_summary.json"
    export_temporal_summary(train_temporal, summary_path)

    example_path = output_path / "temporal_example_rows.csv"
    example_rows = train_temporal[train_temporal[PATIENT_ID_COLUMN].isin(train_temporal[PATIENT_ID_COLUMN].drop_duplicates().head(2))]
    example_rows = example_rows.sort_values([PATIENT_ID_COLUMN, "Hour"]).head(6)
    example_rows.to_csv(example_path, index=False)

    return {
        "selected_features": feature_columns,
        "feature_summary": temporal_summary,
        "metrics": temporal_metrics,
        "pipeline_path": str(pipeline_path),
        "comparison_path": str(temporal_csv_path),
        "metrics_path": str(temporal_json_path),
        "example_path": str(example_path),
    }


if __name__ == "__main__":
    example = pd.DataFrame(
        {
            PATIENT_ID_COLUMN: [1, 1, 1, 2, 2],
            "Hour": [0, 1, 2, 0, 1],
            "HR": [70, 72, 75, 80, 82],
            "WBC": [7.0, 8.5, 9.0, 5.0, 4.5],
            TARGET: [0, 0, 1, 0, 0],
        }
    )
    print(add_temporal_features(example).head().to_string(index=False))
    print(json.dumps(run_temporal_experiment(), indent=2))
