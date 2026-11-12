from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .config import (
    BINARY_TARGET_COLUMN,
    RANDOM_STATE,
    STRICT_BASELINE_EXCLUDED_COLUMNS,
)
from .preprocessing import (
    build_preprocessing_pipeline,
    create_binary_target,
    get_feature_types,
    get_strict_feature_columns,
    load_raw_dataset,
)
from .split_data import prepare_train_test_split


OUTPUT_DIR = Path(__file__).resolve().parents[2] / "models" / "aki_xgboost"
THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
FORBIDDEN_FEATURES = set(STRICT_BASELINE_EXCLUDED_COLUMNS) | {BINARY_TARGET_COLUMN, "Patient_ID"}


@dataclass
class XGBoostConfig:
    n_estimators: int = 300
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = RANDOM_STATE
    n_jobs: int = -1
    validation_size: float = 0.20
    early_stopping_rounds: int = 25
    thresholds: tuple[float, ...] = field(default_factory=lambda: THRESHOLDS)


def validate_feature_columns(feature_columns: list[str]) -> None:
    forbidden = sorted(set(feature_columns).intersection(FORBIDDEN_FEATURES))
    if forbidden:
        raise ValueError(f"Forbidden AKI model features detected: {forbidden}")


def get_final_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_columns = get_strict_feature_columns(df)
    validate_feature_columns(feature_columns)
    return feature_columns


def split_training_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    validation_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X_train,
        y_train,
        test_size=validation_size,
        stratify=y_train,
        random_state=random_state,
    )


def fit_preprocessor(X_train: pd.DataFrame) -> Any:
    numerical, categorical = get_feature_types(X_train, X_train.columns.tolist())
    return build_preprocessing_pipeline(numerical, categorical).fit(X_train)


def transformed_feature_names(preprocessor: Any) -> list[str]:
    transformer = preprocessor.named_steps["preprocessor"]
    return transformer.get_feature_names_out().tolist()


def calculate_scale_pos_weight(y_train: pd.Series | np.ndarray) -> float:
    labels = np.asarray(y_train, dtype=int)
    negative = int(np.sum(labels == 0))
    positive = int(np.sum(labels == 1))
    if negative == 0 or positive == 0:
        raise ValueError("Training labels must contain both classes")
    return float(negative / positive)


def build_model(config: XGBoostConfig, scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        early_stopping_rounds=config.early_stopping_rounds,
    )


def evaluate_probabilities(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "number_of_positive_predictions": int(predictions.sum()),
    }


def threshold_table(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, thresholds: tuple[float, ...] = THRESHOLDS) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        result = evaluate_probabilities(y_true, probabilities, threshold)
        rows.append({key: result[key] for key in ("threshold", "precision", "recall", "f1", "number_of_positive_predictions")})
    return pd.DataFrame(rows)


def select_operating_threshold(table: pd.DataFrame) -> float:
    if table.empty:
        raise ValueError("Cannot select a threshold from an empty table")
    selected = table.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
    return float(selected["threshold"])


def _plot_confusion(metrics: dict[str, Any], plots_dir: Path) -> None:
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No AKI", "AKI"], yticklabels=["No AKI", "AKI"], xlabel="Predicted", ylabel="Actual", title="XGBoost confusion matrix")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix.png")
    plt.close(fig)


def _plot_curves(y_true: pd.Series, probabilities: np.ndarray, metrics: dict[str, Any], plots_dir: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC={metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="Recall", title="AKI XGBoost ROC curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "roc_curve.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR-AUC={metrics['pr_auc']:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="AKI XGBoost precision-recall curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "precision_recall_curve.png")
    plt.close(fig)


def _plot_thresholds(table: pd.DataFrame, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for column in ("precision", "recall", "f1"):
        ax.plot(table["threshold"], table[column], marker="o", label=column)
    ax.set(xlabel="Threshold", ylabel="Score", title="AKI XGBoost threshold analysis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "threshold_comparison.png")
    plt.close(fig)


def _plot_importance(importance: pd.DataFrame, plots_dir: Path) -> None:
    displayed = importance.head(20).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(displayed["feature"], displayed["importance"])
    ax.set(xlabel="Gain importance", title="AKI XGBoost feature importance")
    fig.tight_layout()
    fig.savefig(plots_dir / "feature_importance.png")
    plt.close(fig)


def _load_baseline_comparison() -> pd.DataFrame:
    path = Path(__file__).resolve().parents[2] / "models" / "aki" / "aki_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for name, result in payload["metrics"].items():
        records.append({"Model": name, "Accuracy": result["accuracy"], "Precision": result["precision"], "Recall": result["recall"], "F1": result["f1"], "ROC_AUC": result["roc_auc"], "PR_AUC": result["pr_auc"]})
    return pd.DataFrame(records)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def run_experiment(output_dir: str | Path = OUTPUT_DIR, config: XGBoostConfig | None = None) -> dict[str, Any]:
    config = config or XGBoostConfig()
    output_path = Path(output_dir)
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_dataset()
    feature_columns = get_final_feature_columns(raw_df)
    X_train_full, X_test, y_train_full, y_test, split_summary = prepare_train_test_split()
    validate_feature_columns(X_train_full.columns.tolist())
    X_train, X_validation, y_train, y_validation = split_training_validation(X_train_full, y_train_full, config.validation_size, config.random_state)
    preprocessor = fit_preprocessor(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_validation_processed = preprocessor.transform(X_validation)
    X_test_processed = preprocessor.transform(X_test)
    scale_pos_weight = calculate_scale_pos_weight(y_train)
    model = build_model(config, scale_pos_weight)
    model.fit(X_train_processed, y_train, eval_set=[(X_validation_processed, y_validation)], verbose=False)

    validation_probabilities = model.predict_proba(X_validation_processed)[:, 1]
    test_probabilities = model.predict_proba(X_test_processed)[:, 1]
    thresholds = threshold_table(y_validation, validation_probabilities, config.thresholds)
    selected_threshold = select_operating_threshold(thresholds)
    test_metrics = evaluate_probabilities(y_test, test_probabilities, selected_threshold)

    feature_names = transformed_feature_names(preprocessor)
    importance = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    importance.to_csv(output_path / "feature_importance.csv", index=False)
    thresholds.to_csv(output_path / "threshold_metrics.csv", index=False)
    comparison = _load_baseline_comparison()
    comparison = pd.concat([comparison, pd.DataFrame([{"Model": "XGBoost", "Accuracy": test_metrics["accuracy"], "Precision": test_metrics["precision"], "Recall": test_metrics["recall"], "F1": test_metrics["f1"], "ROC_AUC": test_metrics["roc_auc"], "PR_AUC": test_metrics["pr_auc"]}])], ignore_index=True)
    comparison.to_csv(output_path / "model_comparison.csv", index=False)
    _plot_confusion(test_metrics, plots_dir)
    _plot_curves(y_test, test_probabilities, test_metrics, plots_dir)
    _plot_thresholds(thresholds, plots_dir)
    _plot_importance(importance, plots_dir)

    model_path = output_path / "xgboost_model.json"
    model.get_booster().save_model(model_path)
    joblib.dump(preprocessor, output_path / "preprocessor.joblib")
    best_round = int(model.best_iteration + 1) if model.best_iteration is not None else config.n_estimators
    metadata = {
        "experiment": "isolated AKI XGBoost experiment",
        "dataset": "data/raw/aki/Raw_aki_patient_data.csv",
        "target": BINARY_TARGET_COLUMN,
        "target_definition": "aki_stage == 0 -> 0; aki_stage in {1,2,3} -> 1",
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "transformed_feature_count": len(feature_names),
        "excluded_columns": sorted(FORBIDDEN_FEATURES),
        "split_summary": {**split_summary, "validation_rows": int(len(X_validation)), "train_subset_rows": int(len(X_train)), "patient_id_available": False},
        "preprocessing": "Existing AKI median/mode imputation, scaling, and one-hot encoding fitted on train subset only",
        "class_imbalance": {"scale_pos_weight": scale_pos_weight, "calculated_from": "training subset only"},
        "configuration": _json_safe(asdict(config)),
        "best_iteration": best_round,
        "selected_threshold": selected_threshold,
        "feature_importance_method": "XGBoost gain importance via feature_importances_",
        "limitation": "Feature importance indicates model association/usefulness, not causality or medical significance. This is an academic prototype, not a clinically validated diagnostic system.",
    }
    (output_path / "metadata.json").write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")
    (output_path / "metrics.json").write_text(json.dumps(_json_safe({"test_metrics": test_metrics, "validation_thresholds": thresholds.to_dict(orient="records"), "best_iteration": best_round}), indent=2), encoding="utf-8")
    (output_path / "training_info.json").write_text(json.dumps(_json_safe({"evaluation_metric": "aucpr", "training_rows": len(X_train), "validation_rows": len(X_validation), "test_rows": len(X_test), "best_iteration": best_round}), indent=2), encoding="utf-8")
    return {"metrics": test_metrics, "thresholds": thresholds, "selected_threshold": selected_threshold, "comparison": comparison, "metadata": metadata, "output_dir": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated AKI XGBoost experiment")
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()
    result = run_experiment(config=XGBoostConfig(n_estimators=args.n_estimators))
    print(json.dumps(_json_safe({key: value for key, value in result.items() if key not in {"thresholds", "comparison"}}), indent=2))


if __name__ == "__main__":
    main()