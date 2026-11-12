from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DATA_PATH, PATIENT_ID_COLUMN, RANDOM_STATE, TARGET
from .split_data import load_raw_dataset, patient_level_train_test_split, summarize_split
from .train import patient_level_train_validation_split


LSTM_FEATURES = ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp", "WBC", "Lactate"]
SEQUENCE_LENGTH = 12
DEFAULT_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
DEFAULT_OUTPUT_DIR = Path("models") / "sepsis_lstm"


@dataclass
class LSTMConfig:
    sequence_length: int = SEQUENCE_LENGTH
    lstm_units: int = 64
    dropout: float = 0.30
    dense_units: int = 32
    batch_size: int = 256
    epochs: int = 25
    learning_rate: float = 1e-3
    validation_size: float = 0.20
    early_stopping_patience: int = 4
    random_seed: int = RANDOM_STATE
    thresholds: tuple[float, ...] = field(default_factory=lambda: DEFAULT_THRESHOLDS)
    feature_columns: tuple[str, ...] = field(default_factory=lambda: tuple(LSTM_FEATURES))


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError("The Sepsis LSTM experiment requires tensorflow-cpu. Install dependencies with `pip install -r requirements.txt`.") from exc
    return tf


def set_random_seeds(seed: int = RANDOM_STATE) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        _tensorflow().random.set_seed(seed)
    except ImportError:
        pass


def validate_experiment_columns(df: pd.DataFrame, feature_columns: Iterable[str] = LSTM_FEATURES) -> None:
    feature_columns = list(feature_columns)
    required = {PATIENT_ID_COLUMN, TARGET, "Hour", *feature_columns}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required LSTM columns: {missing}")
    if {PATIENT_ID_COLUMN, TARGET}.intersection(feature_columns):
        raise ValueError("Patient_ID and SepsisLabel cannot be LSTM input features")


def sort_patient_sequences(df: pd.DataFrame) -> pd.DataFrame:
    if PATIENT_ID_COLUMN not in df.columns or "Hour" not in df.columns:
        raise ValueError("Sequence construction requires Patient_ID and Hour")
    sort_columns = [PATIENT_ID_COLUMN, "Hour"]
    if "Unnamed: 0" in df.columns:
        sort_columns.append("Unnamed: 0")
    return df.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def build_preprocessor(feature_columns: Iterable[str] = LSTM_FEATURES) -> Pipeline:
    feature_columns = list(feature_columns)
    if any(column in {PATIENT_ID_COLUMN, TARGET} for column in feature_columns):
        raise ValueError("Identifiers and target are forbidden in the LSTM preprocessor")
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])


def fit_training_preprocessor(train_df: pd.DataFrame, feature_columns: Iterable[str] = LSTM_FEATURES) -> Pipeline:
    feature_columns = list(feature_columns)
    validate_experiment_columns(train_df, feature_columns)
    return build_preprocessor(feature_columns).fit(train_df[feature_columns])


def build_sequences(df: pd.DataFrame, patient_ids: Iterable[Any], preprocessor: Pipeline, sequence_length: int = SEQUENCE_LENGTH, feature_columns: Iterable[str] = LSTM_FEATURES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build left-padded, current-label windows without crossing patient boundaries."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    feature_columns = list(feature_columns)
    validate_experiment_columns(df, feature_columns)
    subset = sort_patient_sequences(df[df[PATIENT_ID_COLUMN].isin(set(patient_ids))].copy())
    if subset.empty:
        return np.empty((0, sequence_length, len(feature_columns)), dtype=np.float32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=object)

    transformed = np.asarray(preprocessor.transform(subset[feature_columns]), dtype=np.float32)
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    sequence_patient_ids: list[Any] = []
    for patient_id, group in subset.groupby(PATIENT_ID_COLUMN, sort=False):
        positions = group.index.to_numpy()
        patient_values = transformed[positions]
        patient_labels = group[TARGET].to_numpy(dtype=np.int32)
        for end in range(len(patient_values)):
            start = max(0, end - sequence_length + 1)
            window = patient_values[start : end + 1]
            padded = np.zeros((sequence_length, len(feature_columns)), dtype=np.float32)
            padded[-len(window) :] = window
            sequences.append(padded)
            labels.append(int(patient_labels[end]))
            sequence_patient_ids.append(patient_id)
    return np.stack(sequences), np.asarray(labels, dtype=np.int32), np.asarray(sequence_patient_ids)


def calculate_class_weights(labels: np.ndarray) -> dict[int, float]:
    labels = np.asarray(labels, dtype=np.int32)
    negative = int(np.sum(labels == 0))
    positive = int(np.sum(labels == 1))
    if not negative or not positive:
        raise ValueError("Training labels must contain both classes for class weighting")
    total = negative + positive
    return {0: total / (2.0 * negative), 1: total / (2.0 * positive)}


def build_lstm_model(input_shape: tuple[int, int], config: LSTMConfig) -> Any:
    tf = _tensorflow()
    inputs = tf.keras.Input(shape=input_shape, name="patient_sequence")
    masked = tf.keras.layers.Masking(mask_value=0.0, name="padding_mask")(inputs)
    encoded = tf.keras.layers.LSTM(config.lstm_units, name="lstm")(masked)
    dropped = tf.keras.layers.Dropout(config.dropout, name="dropout")(encoded)
    dense = tf.keras.layers.Dense(config.dense_units, activation="relu", name="dense")(dropped)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="sepsis_probability")(dense)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="sepsis_lstm")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate), loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="roc_auc"), tf.keras.metrics.AUC(curve="PR", name="pr_auc")])
    return model


def evaluate_probabilities(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    predictions = (probabilities >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(y_true)) > 1
    return {"threshold": float(threshold), "accuracy": float(accuracy_score(y_true, predictions)), "precision": float(precision_score(y_true, predictions, zero_division=0)), "recall": float(recall_score(y_true, predictions, zero_division=0)), "f1": float(f1_score(y_true, predictions, zero_division=0)), "roc_auc": float(roc_auc_score(y_true, probabilities)) if has_both_classes else float("nan"), "pr_auc": float(average_precision_score(y_true, probabilities)) if has_both_classes else float("nan"), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "number_of_positive_predictions": int(predictions.sum())}


def threshold_table(y_true: np.ndarray, probabilities: np.ndarray, thresholds: Iterable[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        result = evaluate_probabilities(y_true, probabilities, float(threshold))
        rows.append({key: result[key] for key in ("threshold", "precision", "recall", "f1", "number_of_positive_predictions")})
    return pd.DataFrame(rows)


def select_operating_threshold(table: pd.DataFrame) -> float:
    if table.empty:
        raise ValueError("Threshold table is empty")
    selected = table.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
    return float(selected["threshold"])


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


def _plot_training_history(history: dict[str, list[float]], plots_dir: Path) -> None:
    for metric, filename, title in [("loss", "loss_curve.png", "LSTM training and validation loss"), ("pr_auc", "pr_auc_curve.png", "LSTM training and validation PR-AUC")]:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(history.get(metric, []), label="training")
        ax.plot(history.get(f"val_{metric}", []), label="validation")
        ax.set(xlabel="Epoch", ylabel=metric, title=title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / filename)
        plt.close(fig)


def _plot_test_curves(y_true: np.ndarray, probabilities: np.ndarray, metrics: dict[str, Any], plots_dir: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC={metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="Recall", title="Sepsis LSTM ROC curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "roc_curve.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label=f"PR-AUC={metrics['pr_auc']:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Sepsis LSTM precision-recall curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "precision_recall_curve.png")
    plt.close(fig)
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"], xlabel="Predicted", ylabel="Actual", title="Sepsis LSTM confusion matrix")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix.png")
    plt.close(fig)


def _plot_thresholds(table: pd.DataFrame, plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for column in ("precision", "recall", "f1"):
        ax.plot(table["threshold"], table[column], marker="o", label=column)
    ax.set(xlabel="Probability threshold", ylabel="Score", title="Sepsis LSTM threshold analysis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "threshold_comparison.png")
    plt.close(fig)


def _load_existing_metrics(project_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    baseline_path = project_root / "models" / "sepsis_metrics.json"
    temporal_path = project_root / "models" / "temporal" / "temporal_metrics.json"
    if baseline_path.exists():
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_records = payload if isinstance(payload, list) else payload.get("baseline_model_metrics", [])
        baseline = next((row for row in baseline_records if row.get("model") == "LogisticRegression"), None)
        if baseline:
            records.append({key: baseline.get(key) for key in ("model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")})
    if temporal_path.exists():
        payload = json.loads(temporal_path.read_text(encoding="utf-8"))
        temporal = payload.get("temporal_model", {})
        if temporal:
            records.append({"model": "Temporal Logistic Regression", **{key: temporal.get(key) for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")}})
    return records


def _comparison_summary(comparison: pd.DataFrame) -> dict[str, str]:
    return {metric: str(comparison.loc[comparison[metric].astype(float).idxmax(), "model"]) for metric in ("recall", "f1", "pr_auc", "roc_auc") if not comparison.empty and comparison[metric].notna().any()}


def run_lstm_experiment(output_dir: str | Path = DEFAULT_OUTPUT_DIR, data_path: str | Path = DATA_PATH, config: LSTMConfig | None = None) -> dict[str, Any]:
    config = config or LSTMConfig()
    if config.sequence_length < 1 or config.epochs < 1:
        raise ValueError("sequence_length and epochs must be positive")
    set_random_seeds(config.random_seed)
    output_path = Path(output_dir)
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    df = load_raw_dataset(Path(data_path))
    validate_experiment_columns(df, config.feature_columns)
    _, _, _, _, train_ids, test_ids = patient_level_train_test_split(df, random_state=config.random_seed)
    train_ids, validation_ids = patient_level_train_validation_split(df, train_ids, validation_size=config.validation_size, random_state=config.random_seed)
    if set(train_ids).intersection(test_ids) or set(train_ids).intersection(validation_ids) or set(validation_ids).intersection(test_ids):
        raise RuntimeError("Patient overlap detected among train, validation, and test splits")
    train_df = df[df[PATIENT_ID_COLUMN].isin(train_ids)]
    preprocessor = fit_training_preprocessor(train_df, config.feature_columns)
    X_train, y_train, _ = build_sequences(df, train_ids, preprocessor, config.sequence_length, config.feature_columns)
    X_validation, y_validation, _ = build_sequences(df, validation_ids, preprocessor, config.sequence_length, config.feature_columns)
    X_test, y_test, _ = build_sequences(df, test_ids, preprocessor, config.sequence_length, config.feature_columns)
    class_weights = calculate_class_weights(y_train)
    model = build_lstm_model((config.sequence_length, len(config.feature_columns)), config)
    tf = _tensorflow()
    checkpoint_path = output_path / "best_model.keras"
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=config.early_stopping_patience, restore_best_weights=True), tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_pr_auc", mode="max", save_best_only=True)]
    history_object = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), class_weight=class_weights, batch_size=config.batch_size, epochs=config.epochs, callbacks=callbacks, verbose=2)
    history = {key: [float(value) for value in values] for key, values in history_object.history.items()}
    model.save(output_path / "sepsis_lstm.keras")
    probabilities_validation = model.predict(X_validation, batch_size=config.batch_size, verbose=0).reshape(-1)
    probabilities_test = model.predict(X_test, batch_size=config.batch_size, verbose=0).reshape(-1)
    thresholds = threshold_table(y_validation, probabilities_validation, config.thresholds)
    selected_threshold = select_operating_threshold(thresholds)
    test_metrics = evaluate_probabilities(y_test, probabilities_test, selected_threshold)
    thresholds.to_csv(output_path / "threshold_metrics.csv", index=False)
    _plot_training_history(history, plots_dir)
    _plot_test_curves(y_test, probabilities_test, test_metrics, plots_dir)
    _plot_thresholds(thresholds, plots_dir)
    project_root = Path(__file__).resolve().parents[2]
    comparison_records = _load_existing_metrics(project_root)
    comparison_records.append({"model": "LSTM", **{key: test_metrics[key] for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")}})
    comparison = pd.DataFrame(comparison_records, columns=["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"])
    comparison.to_csv(output_path / "model_comparison.csv", index=False)
    split_summary = summarize_split(df, train_ids, test_ids)
    split_summary.update({"validation_patients": len(validation_ids), "validation_observations": len(df[df[PATIENT_ID_COLUMN].isin(validation_ids)]), "patient_overlap_train_validation": len(set(train_ids).intersection(validation_ids)), "patient_overlap_validation_test": len(set(validation_ids).intersection(test_ids)), "sequence_train_samples": len(X_train), "sequence_validation_samples": len(X_validation), "sequence_test_samples": len(X_test)})
    metadata = {"experiment": "experimental Sepsis LSTM sequence model", "features": list(config.feature_columns), "target": TARGET, "label_definition": "Existing SepsisLabel at the current observation; no future-horizon relabeling", "sequence_length": config.sequence_length, "padding": "left zero padding after train-fitted scaling with Keras Masking", "preprocessing": "training-patient median imputation followed by StandardScaler; applied unchanged to validation/test", "class_weights": class_weights, "selected_threshold": selected_threshold, "configuration": _json_safe(asdict(config)), "architecture": {"masking": True, "lstm_units": config.lstm_units, "dropout": config.dropout, "dense_units": config.dense_units, "output": "sigmoid"}, "split_summary": split_summary, "best_model": str(checkpoint_path), "comparison_best_models": _comparison_summary(comparison), "limitation": "Academic research prototype; dataset performance does not establish real-world clinical effectiveness or clinical validity."}
    (output_path / "metadata.json").write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")
    (output_path / "metrics.json").write_text(json.dumps(_json_safe({"test_metrics": test_metrics, "validation_thresholds": thresholds.to_dict(orient="records"), "comparison_best_models": _comparison_summary(comparison)}), indent=2), encoding="utf-8")
    (output_path / "training_history.json").write_text(json.dumps(_json_safe(history), indent=2), encoding="utf-8")
    (output_path / "split_summary.json").write_text(json.dumps(_json_safe(split_summary), indent=2), encoding="utf-8")
    joblib.dump(preprocessor, output_path / "preprocessor.joblib")
    joblib.dump({"feature_columns": list(config.feature_columns), "sequence_length": config.sequence_length, "threshold": selected_threshold}, output_path / "inference_metadata.joblib")
    return {"metrics": test_metrics, "selected_threshold": selected_threshold, "comparison": comparison, "split_summary": split_summary, "output_dir": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the experimental patient-level Sepsis LSTM")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    args = parser.parse_args()
    result = run_lstm_experiment(config=LSTMConfig(epochs=args.epochs, batch_size=args.batch_size, sequence_length=args.sequence_length))
    print(json.dumps(_json_safe({key: value for key, value in result.items() if key != "comparison"}), indent=2))


if __name__ == "__main__":
    main()