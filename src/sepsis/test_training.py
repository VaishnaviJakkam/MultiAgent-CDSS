import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from src.sepsis.train import METADATA_PATH, METRICS_CSV, METRICS_JSON, PIPELINE_PATH, run_training
from src.sepsis.config import TARGET
from src.sepsis.split_data import load_raw_dataset, patient_level_train_test_split


def test_training_pipeline():
    df = load_raw_dataset()
    X_train, X_test, y_train, y_test, train_ids, test_ids = patient_level_train_test_split(df)

    assert len(train_ids) > 0
    assert len(test_ids) > 0
    assert set(train_ids) & set(test_ids) == set()
    assert TARGET in y_train.name

    result = run_training()
    assert result["selected_model"] in ["LogisticRegression", "DecisionTree", "RandomForest"]
    assert Path(result["pipeline_path"]).exists()
    assert Path(result["metrics_json"]).exists()
    assert Path(result["metrics_csv"]).exists()
    assert Path(result["metadata_path"]).exists()

    pipeline = joblib.load(result["pipeline_path"])
    assert "preprocessor" in pipeline and "model" in pipeline
    assert pipeline["target"] == TARGET

    sample = X_test.head(5)
    transformed = pipeline["preprocessor"].transform(sample)
    preds = pipeline["model"].predict(transformed)
    probas = pipeline["model"].predict_proba(transformed)[:, 1]

    assert len(preds) == 5
    assert len(probas) == 5
    assert all(0.0 <= p <= 1.0 for p in probas)

    metrics_df = pd.read_csv(METRICS_CSV)
    assert set(metrics_df.columns) >= {"model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
    assert not metrics_df.empty

    loaded = pd.read_json(METRICS_JSON)
    assert not loaded.empty

    print("Training pipeline test passed")


if __name__ == "__main__":
    test_training_pipeline()
