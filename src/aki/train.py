from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import (
    BINARY_TARGET_COLUMN,
    POTENTIAL_TARGET_DEFINING_COLUMNS,
    RANDOM_STATE,
    STRICT_BASELINE_EXCLUDED_COLUMNS,
    TARGET_COLUMN,
)
from .preprocessing import (
    build_preprocessing_pipeline,
    get_strict_feature_columns,
    load_raw_dataset,
)
from .split_data import prepare_train_test_split

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "aki"
PIPELINE_PATH = MODELS_DIR / "aki_pipeline.pkl"
METRICS_CSV_PATH = MODELS_DIR / "aki_model_comparison.csv"
METRICS_JSON_PATH = MODELS_DIR / "aki_metrics.json"
METADATA_JSON_PATH = MODELS_DIR / "aki_metadata.json"
CONFUSION_MATRIX_FILES = {
    "LogisticRegression": MODELS_DIR / "aki_logistic_regression_confusion_matrix.png",
    "DecisionTree": MODELS_DIR / "aki_decision_tree_confusion_matrix.png",
    "RandomForest": MODELS_DIR / "aki_random_forest_confusion_matrix.png",
}


def ensure_output_dir() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_baseline_models() -> dict[str, Any]:
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=200,
            random_state=RANDOM_STATE,
        ),
        "DecisionTree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_model(name: str, model: Any, X: np.ndarray, y: pd.Series) -> dict[str, Any]:
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X)
    else:
        y_proba = np.zeros_like(y_pred, dtype=float)

    if len(np.unique(y)) > 1:
        roc_auc = float(roc_auc_score(y, y_proba))
        pr_auc = float(average_precision_score(y, y_proba))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return {
        "model": name,
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
    }


def plot_confusion_matrix(name: str, cm: dict[str, int]) -> None:
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"{name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0 = No AKI", "1 = AKI"])
    ax.set_yticklabels(["0 = No AKI", "1 = AKI"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(matrix[i, j], "d"), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_FILES[name])
    plt.close(fig)


def select_best_model(results: dict[str, dict[str, Any]]) -> str:
    sorted_models = sorted(
        results.items(),
        key=lambda item: (
            item[1]["recall"],
            item[1]["pr_auc"],
            item[1]["f1"],
            item[1]["precision"],
            item[1]["roc_auc"],
            item[1]["accuracy"],
        ),
        reverse=True,
    )
    return sorted_models[0][0]


def build_pipeline(preprocessor: Pipeline, model: Any) -> Pipeline:
    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])


def save_model_pipeline(pipeline: Pipeline) -> None:
    ensure_output_dir()
    joblib.dump(pipeline, PIPELINE_PATH)


def save_metrics(results: dict[str, dict[str, Any]], selected_model: str, threshold: float = 0.50) -> None:
    records = []
    for name, result in results.items():
        records.append(
            {
                "Model": name,
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1": result["f1"],
                "ROC_AUC": result["roc_auc"],
                "PR_AUC": result["pr_auc"],
            }
        )
    pd.DataFrame(records).to_csv(METRICS_CSV_PATH, index=False)

    metrics_payload = {
        "selected_model": selected_model,
        "threshold": threshold,
        "random_state": RANDOM_STATE,
        "metrics": {name: {k: v for k, v in result.items() if k not in {"y_pred", "y_proba"}} for name, result in results.items()},
        "feature_count": len(get_strict_feature_columns(load_raw_dataset())),
        "feature_names": get_strict_feature_columns(load_raw_dataset()),
        "target": BINARY_TARGET_COLUMN,
        "target_definition": f"{TARGET_COLUMN} == 0 -> 0; {TARGET_COLUMN} in {{1,2,3}} -> 1",
        "excluded_columns": STRICT_BASELINE_EXCLUDED_COLUMNS,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)


def save_metadata(selected_model: str, split_summary: dict[str, Any]) -> None:
    metadata = {
        "dataset_rows": split_summary["total_rows"],
        "train_rows": split_summary["train_rows"],
        "test_rows": split_summary["test_rows"],
        "target": BINARY_TARGET_COLUMN,
        "target_mapping": {"aki_stage == 0": 0, "aki_stage in {1,2,3}": 1},
        "model_selected": selected_model,
        "feature_names": get_strict_feature_columns(load_raw_dataset()),
        "excluded_columns": STRICT_BASELINE_EXCLUDED_COLUMNS,
        "preprocessing_strategy": {
            "numerical": "SimpleImputer(strategy='median') + StandardScaler",
            "categorical": "SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore')",
        },
        "threshold": 0.50,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "note": "This is a strict leakage-aware baseline. Creatinine and urine-output variables that may contribute directly to AKI staging were excluded.",
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def run_baseline_training() -> dict[str, Any]:
    ensure_output_dir()
    X_train, X_test, y_train, y_test, split_summary = prepare_train_test_split()
    assert BINARY_TARGET_COLUMN not in X_train.columns, "Target must not be included in training features"
    for excluded_column in STRICT_BASELINE_EXCLUDED_COLUMNS:
        assert excluded_column not in X_train.columns, f"Excluded column {excluded_column} entered X_train"
        assert excluded_column not in X_test.columns, f"Excluded column {excluded_column} entered X_test"

    numerical_features = [col for col in X_train.columns if np.issubdtype(X_train[col].dtype, np.number)]
    categorical_features = [col for col in X_train.columns if not np.issubdtype(X_train[col].dtype, np.number)]

    preprocessor = build_preprocessing_pipeline(numerical_features, categorical_features)
    fitted_preprocessor = preprocessor.fit(X_train)
    X_train_processed = fitted_preprocessor.transform(X_train)
    X_test_processed = fitted_preprocessor.transform(X_test)

    models = build_baseline_models()
    trained_models: dict[str, Any] = {}
    results: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        trained_models[name] = model
        results[name] = evaluate_model(name, model, X_test_processed, y_test)
        plot_confusion_matrix(name, results[name]["confusion_matrix"])

    selected_model_name = select_best_model(results)
    selected_model = trained_models[selected_model_name]
    pipeline = build_pipeline(fitted_preprocessor, selected_model)
    save_model_pipeline(pipeline)
    save_metrics(results, selected_model_name)
    save_metadata(selected_model_name, split_summary)

    return {
        "split_summary": split_summary,
        "results": results,
        "selected_model": selected_model_name,
        "pipeline_path": str(PIPELINE_PATH),
        "metrics_json_path": str(METRICS_JSON_PATH),
        "metadata_json_path": str(METADATA_JSON_PATH),
        "comparison_csv_path": str(METRICS_CSV_PATH),
        "confusion_matrix_paths": {name: str(path) for name, path in CONFUSION_MATRIX_FILES.items()},
    }


if __name__ == "__main__":
    summary = run_baseline_training()
    print(json.dumps(summary, indent=2))
