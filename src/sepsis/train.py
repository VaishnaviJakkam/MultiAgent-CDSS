from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
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
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

try:
    from .config import FEATURE_COLUMNS, PATIENT_ID_COLUMN, RANDOM_STATE, TARGET
    from .preprocessing import build_preprocessing_pipeline, transform_features
    from .split_data import load_raw_dataset, patient_level_train_test_split, summarize_split
except ImportError:  # pragma: no cover - support direct script execution
    from src.sepsis.config import FEATURE_COLUMNS, PATIENT_ID_COLUMN, RANDOM_STATE, TARGET
    from src.sepsis.preprocessing import build_preprocessing_pipeline, transform_features
    from src.sepsis.split_data import load_raw_dataset, patient_level_train_test_split, summarize_split

OUTPUT_DIR = Path("models")
EVALUATION_DIR = OUTPUT_DIR / "evaluation"
PIPELINE_PATH = OUTPUT_DIR / "sepsis_pipeline.pkl"
METADATA_PATH = OUTPUT_DIR / "sepsis_metadata.json"
METRICS_JSON = OUTPUT_DIR / "sepsis_metrics.json"
METRICS_CSV = OUTPUT_DIR / "sepsis_model_comparison.csv"
THRESHOLD_METRICS_CSV = EVALUATION_DIR / "threshold_metrics.csv"
THRESHOLD_PRECISION_RECALL_PLOT = EVALUATION_DIR / "threshold_precision_recall.png"
THRESHOLD_F1_PLOT = EVALUATION_DIR / "threshold_f1.png"
FINAL_CONFUSION_MATRIX_PLOT = EVALUATION_DIR / "final_confusion_matrix.png"


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


def train_models(X_train: np.ndarray, y_train: pd.Series) -> dict[str, Any]:
    models: dict[str, Any] = {}

    models["LogisticRegression"] = LogisticRegression(
        class_weight="balanced",
        solver="saga",
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    models["DecisionTree"] = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    training_times: dict[str, float] = {}
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        training_times[name] = time.perf_counter() - start
    models["training_times"] = training_times
    return models


def evaluate_model(name: str, model: Any, X: np.ndarray, y: pd.Series) -> dict[str, Any]:
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X)
    else:
        y_proba = np.zeros_like(y_pred, dtype=float)

    roc_auc = roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else float("nan")
    pr_auc = average_precision_score(y, y_proba) if len(np.unique(y)) > 1 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    return {
        "model": name,
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def plot_confusion_matrix(name: str, cm: dict[str, int]) -> None:
    labels = ["Negative", "Positive"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    file_name = {
        "LogisticRegression": "logistic_regression_confusion_matrix.png",
        "DecisionTree": "decision_tree_confusion_matrix.png",
        "RandomForest": "random_forest_confusion_matrix.png",
    }.get(name, f"{name.lower()}_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"{name} Confusion Matrix",
    )
    fmt = "d"
    thresh = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color="white" if matrix[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(EVALUATION_DIR / file_name)
    plt.close(fig)


def plot_curves(results: dict[str, dict[str, Any]], y_test: pd.Series) -> None:
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))

    for name, result in results.items():
        y_proba = result["y_proba"]
        if len(np.unique(y_test)) <= 1:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={result['roc_auc']:.3f})")
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ax_pr.plot(recall, precision, label=f"{name} (AP={result['pr_auc']:.3f})")

    ax_roc.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve Comparison")
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(EVALUATION_DIR / "roc_curve_comparison.png")
    plt.close(fig_roc)

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve Comparison")
    ax_pr.legend(loc="lower left")
    fig_pr.tight_layout()
    fig_pr.savefig(EVALUATION_DIR / "pr_curve_comparison.png")
    plt.close(fig_pr)


def select_best_model(results: dict[str, dict[str, Any]]) -> str:
    best_name = None
    best_score = -1.0
    best_tuple: tuple[float, float, float] | None = None

    for name, res in results.items():
        tuple_score = (res["recall"], res["f1"], res["pr_auc"])
        score = res["pr_auc"]
        if best_name is None or score > best_score or (np.isclose(score, best_score) and tuple_score > (best_tuple[0], best_tuple[1], best_tuple[2])):
            best_score = score
            best_name = name
            best_tuple = tuple_score

    return best_name or "RandomForest"


def save_metrics(results: dict[str, dict[str, Any]]) -> None:
    records = []
    for name, res in results.items():
        record = {
            "model": name,
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1": res["f1"],
            "roc_auc": res["roc_auc"],
            "pr_auc": res["pr_auc"],
            "tn": res["confusion_matrix"]["tn"],
            "fp": res["confusion_matrix"]["fp"],
            "fn": res["confusion_matrix"]["fn"],
            "tp": res["confusion_matrix"]["tp"],
        }
        records.append(record)
    pd.DataFrame(records).to_csv(METRICS_CSV, index=False)
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def save_pipeline(preprocessor: Any, model: Any, model_name: str, selected_threshold: float | None = None) -> None:
    payload = {
        "preprocessor": preprocessor,
        "model": model,
        "model_name": model_name,
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "selected_threshold": selected_threshold,
    }
    joblib.dump(payload, PIPELINE_PATH)
    return None


def save_metadata(selected_model: str, metrics: dict[str, dict[str, Any]], split_summary: dict[str, Any]) -> None:
    metadata = {
        "selected_model": selected_model,
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "split_summary": split_summary,
        "evaluation_metrics": {
            name: {
                k: v
                for k, v in res.items()
                if k not in {"y_pred", "y_proba"}
            }
            for name, res in metrics.items()
        },
        "preprocessing": {
            "numeric_imputation": "median",
            "categorical_imputation": "most_frequent",
            "scaler": "StandardScaler",
            "categorical_encoding": "OneHotEncoder(drop='if_binary', handle_unknown='ignore')",
        },
        "model_defaults": {
            "LogisticRegression": {"class_weight": "balanced", "solver": "saga", "max_iter": 1000, "random_state": RANDOM_STATE},
            "DecisionTree": {"class_weight": "balanced", "random_state": RANDOM_STATE},
            "RandomForest": {"n_estimators": 100, "class_weight": "balanced", "random_state": RANDOM_STATE, "n_jobs": -1},
        },
        "training_timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def prepare_serializable_summary(training_times: dict[str, float], results: dict[str, dict[str, Any]], selected_model_name: str, split_summary: dict[str, Any]) -> dict[str, Any]:
    serializable_results: dict[str, dict[str, Any]] = {}
    for name, result in results.items():
        serializable_result = {}
        for key, value in result.items():
            if key in {"y_pred", "y_proba"} and isinstance(value, np.ndarray):
                serializable_result[key] = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                serializable_result[key] = value.item()
            else:
                serializable_result[key] = value
        serializable_results[name] = serializable_result

    return {
        "split_summary": split_summary,
        "training_times": {k: float(v) for k, v in training_times.items()},
        "results": serializable_results,
        "selected_model": selected_model_name,
        "pipeline_path": str(PIPELINE_PATH),
        "metrics_json": str(METRICS_JSON),
        "metrics_csv": str(METRICS_CSV),
        "metadata_path": str(METADATA_PATH),
    }


def patient_level_train_validation_split(df: pd.DataFrame, train_ids: pd.Index, validation_size: float = 0.20, random_state: int = RANDOM_STATE) -> tuple[pd.Index, pd.Index]:
    train_mask = df[PATIENT_ID_COLUMN].isin(train_ids)
    train_patient_labels = df.loc[train_mask].groupby(PATIENT_ID_COLUMN)[TARGET].max()
    available_ids = train_patient_labels.index.to_numpy()
    available_labels = train_patient_labels.to_numpy()

    train_part, valid_part = train_test_split(
        available_ids,
        test_size=validation_size,
        random_state=random_state,
        stratify=available_labels,
    )

    return pd.Index(sorted(train_part)), pd.Index(sorted(valid_part))


def evaluate_threshold(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=float)
    y_true_array = np.asarray(y_true, dtype=int)
    pred = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_array, pred, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_true_array, pred)
    precision = precision_score(y_true_array, pred, zero_division=0)
    recall = recall_score(y_true_array, pred, zero_division=0)
    f1 = f1_score(y_true_array, pred, zero_division=0)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_count": int(fp),
        "false_negative_count": int(fn),
        "true_positive_count": int(tp),
        "true_negative_count": int(tn),
    }


def threshold_summary_table(probabilities: np.ndarray, y_true: pd.Series | np.ndarray, thresholds: list[float]) -> list[dict[str, Any]]:
    return [evaluate_threshold(y_true, probabilities, threshold) for threshold in thresholds]


def select_threshold_from_validation(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("No threshold results were generated")

    best = max(
        results,
        key=lambda item: (item["f1"], item["recall"], item["precision"]),
    )
    return best


def save_threshold_metrics(results: list[dict[str, Any]]) -> None:
    pd.DataFrame(results).to_csv(THRESHOLD_METRICS_CSV, index=False)


def plot_threshold_analysis(results: list[dict[str, Any]], probabilities: np.ndarray, y_true: pd.Series | np.ndarray) -> None:
    df = pd.DataFrame(results)
    thresholds = df["threshold"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].plot(thresholds, df["precision"], marker="o", label="Precision")
    axes[0, 0].set_title("Precision vs threshold")
    axes[0, 0].set_xlabel("Threshold")
    axes[0, 0].set_ylabel("Precision")
    axes[0, 0].grid(True)

    axes[0, 1].plot(thresholds, df["recall"], marker="o", label="Recall")
    axes[0, 1].set_title("Recall vs threshold")
    axes[0, 1].set_xlabel("Threshold")
    axes[0, 1].set_ylabel("Recall")
    axes[0, 1].grid(True)

    axes[1, 0].plot(thresholds, df["f1"], marker="o", label="F1")
    axes[1, 0].set_title("F1 vs threshold")
    axes[1, 0].set_xlabel("Threshold")
    axes[1, 0].set_ylabel("F1")
    axes[1, 0].grid(True)

    precision_curve, recall_curve, _ = precision_recall_curve(np.asarray(y_true, dtype=int), probabilities)
    axes[1, 1].plot(recall_curve, precision_curve, marker="o", label="PR curve")
    axes[1, 1].set_title("Precision-Recall curve")
    axes[1, 1].set_xlabel("Recall")
    axes[1, 1].set_ylabel("Precision")
    axes[1, 1].grid(True)

    fig.tight_layout()
    fig.savefig(THRESHOLD_PRECISION_RECALL_PLOT)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(thresholds, df["f1"], marker="o", label="F1")
    ax2.set_title("Validation F1 by threshold")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("F1 score")
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig(THRESHOLD_F1_PLOT)
    plt.close(fig2)


def update_metrics_json(baseline_metrics: list[dict[str, Any]], threshold_analysis: dict[str, Any]) -> None:
    if METRICS_JSON.exists():
        try:
            existing = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
        if isinstance(existing, list):
            baseline_payload = existing
        else:
            baseline_payload = existing.get("baseline_model_metrics", [])
    else:
        baseline_payload = []

    payload = {
        "baseline_model_metrics": baseline_payload if baseline_payload else baseline_metrics,
        "threshold_analysis": threshold_analysis,
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_training() -> dict[str, Any]:
    ensure_output_dirs()
    df = load_raw_dataset()
    X_train, X_test, y_train, y_test, train_ids, test_ids = patient_level_train_test_split(df)
    split_summary = summarize_split(df, train_ids, test_ids)

    preprocessor = build_preprocessing_pipeline()
    fitted_preprocessor = preprocessor.fit(X_train)
    X_train_processed = transform_features(fitted_preprocessor, X_train)
    X_test_processed = transform_features(fitted_preprocessor, X_test)

    models = train_models(X_train_processed, y_train)
    training_times = models.pop("training_times")

    results: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        results[name] = evaluate_model(name, model, X_test_processed, y_test)
        plot_confusion_matrix(name, results[name]["confusion_matrix"])

    plot_curves(results, y_test)
    save_metrics(results)

    selected_model_name = select_best_model(results)
    selected_model = models[selected_model_name]
    save_pipeline(fitted_preprocessor, selected_model, selected_model_name)
    save_metadata(selected_model_name, results, split_summary)

    return prepare_serializable_summary(training_times, results, selected_model_name, split_summary)


def run_threshold_analysis() -> dict[str, Any]:
    ensure_output_dirs()
    df = load_raw_dataset()
    X_train_full, X_test, y_train_full, y_test, train_ids, test_ids = patient_level_train_test_split(df)
    split_summary = summarize_split(df, train_ids, test_ids)

    train_validation_ids, val_ids = patient_level_train_validation_split(df, train_ids, validation_size=0.20, random_state=RANDOM_STATE)
    validation_train_mask = df[PATIENT_ID_COLUMN].isin(train_validation_ids)
    validation_val_mask = df[PATIENT_ID_COLUMN].isin(val_ids)

    X_train_validation = df.loc[validation_train_mask, FEATURE_COLUMNS].copy()
    y_train_validation = df.loc[validation_train_mask, TARGET].copy()
    X_val = df.loc[validation_val_mask, FEATURE_COLUMNS].copy()
    y_val = df.loc[validation_val_mask, TARGET].copy()

    preprocessor = build_preprocessing_pipeline()
    fitted_preprocessor = preprocessor.fit(X_train_validation)
    X_train_valid_processed = transform_features(fitted_preprocessor, X_train_validation)
    X_val_processed = transform_features(fitted_preprocessor, X_val)

    logistic_model = LogisticRegression(
        class_weight="balanced",
        solver="saga",
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    logistic_model.fit(X_train_valid_processed, y_train_validation)
    val_probabilities = logistic_model.predict_proba(X_val_processed)[:, 1]

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    threshold_results = threshold_summary_table(val_probabilities, y_val, thresholds)
    selected = select_threshold_from_validation(threshold_results)

    save_threshold_metrics(threshold_results)
    plot_threshold_analysis(threshold_results, val_probabilities, y_val)

    final_preprocessor = build_preprocessing_pipeline()
    final_fitted_preprocessor = final_preprocessor.fit(X_train_full)
    X_train_full_processed = transform_features(final_fitted_preprocessor, X_train_full)
    X_test_processed = transform_features(final_fitted_preprocessor, X_test)
    final_model = LogisticRegression(
        class_weight="balanced",
        solver="saga",
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final_model.fit(X_train_full_processed, y_train_full)
    test_probabilities_50 = final_model.predict_proba(X_test_processed)[:, 1]
    test_threshold_50 = evaluate_threshold(y_test, test_probabilities_50, 0.50)
    selected_threshold = float(selected["threshold"])
    test_probabilities_selected = final_model.predict_proba(X_test_processed)[:, 1]
    test_threshold_selected = evaluate_threshold(y_test, test_probabilities_selected, selected_threshold)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(np.array([[test_threshold_50["tn"], test_threshold_50["fp"]], [test_threshold_50["fn"], test_threshold_50["tp"]]]), cmap="Blues")
    ax.set_title("Original test confusion matrix at 0.50 threshold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])
    fig.tight_layout()
    fig.savefig(FINAL_CONFUSION_MATRIX_PLOT)
    plt.close(fig)

    save_pipeline(final_fitted_preprocessor, final_model, "LogisticRegression", selected_threshold)

    validation_summary = {
        "validation_train_patients": len(train_validation_ids),
        "validation_patients": len(val_ids),
        "validation_observations": int(validation_val_mask.sum()),
        "validation_prevalence": float(y_val.mean()),
        "thresholds_evaluated": thresholds,
        "selected_threshold": selected_threshold,
        "selection_criterion": "maximize validation F1-score, then preserve recall as tie-breaker",
        "selected_validation_metrics": selected,
    }

    metadata = {
        "selected_model": "LogisticRegression",
        "target": TARGET,
        "feature_columns": FEATURE_COLUMNS,
        "selected_threshold": selected_threshold,
        "threshold_selection_criterion": validation_summary["selection_criterion"],
        "preprocessing_description": {
            "numeric_imputation": "median",
            "categorical_imputation": "most_frequent",
            "scaler": "StandardScaler",
            "categorical_encoding": "OneHotEncoder(drop='if_binary', handle_unknown='ignore')",
        },
        "validation_summary": validation_summary,
        "training_split_summary": split_summary,
        "baseline_test_metrics_at_0_50": test_threshold_50,
        "selected_threshold_test_metrics": test_threshold_selected,
        "config": {
            "class_weight": "balanced",
            "solver": "saga",
            "max_iter": 1000,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
        "training_timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    legacy_baseline = []
    if METRICS_JSON.exists():
        try:
            legacy = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
            if isinstance(legacy, list):
                legacy_baseline = legacy
            elif isinstance(legacy, dict):
                legacy_baseline = legacy.get("baseline_model_metrics", [])
        except json.JSONDecodeError:
            legacy_baseline = []

    threshold_analysis = {
        "selected_threshold": selected_threshold,
        "selection_criterion": validation_summary["selection_criterion"],
        "thresholds_evaluated": thresholds,
        "threshold_metrics": threshold_results,
        "selected_validation_metrics": selected,
        "baseline_test_metrics_at_0_50": test_threshold_50,
        "final_test_metrics_at_selected_threshold": test_threshold_selected,
        "validation_split_info": {
            "validation_patients": len(val_ids),
            "validation_observations": int(validation_val_mask.sum()),
            "validation_prevalence": float(y_val.mean()),
            "train_patients": len(train_validation_ids),
        },
    }

    update_metrics_json(legacy_baseline, threshold_analysis)

    return {
        "selected_threshold": selected_threshold,
        "validation_split": {
            "training_patients": len(train_validation_ids),
            "validation_patients": len(val_ids),
            "validation_observations": int(validation_val_mask.sum()),
            "validation_prevalence": float(y_val.mean()),
        },
        "threshold_results": threshold_results,
        "selected_validation_metrics": selected,
        "baseline_test_metrics_at_0_50": test_threshold_50,
        "final_test_metrics_at_selected_threshold": test_threshold_selected,
        "model_path": str(PIPELINE_PATH),
        "metadata_path": str(METADATA_PATH),
    }


if __name__ == "__main__":
    output = run_threshold_analysis()
    print(json.dumps(output, indent=2))
