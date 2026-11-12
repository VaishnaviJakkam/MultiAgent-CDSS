# Experimental Sepsis LSTM

This document describes a separate academic experiment. It does not replace the existing baseline or temporal Logistic Regression pipeline, saved models, or Sepsis Detection Agent.

## Purpose

The experiment tests whether a lightweight recurrent model can learn from recent patient observations directly, instead of relying only on the existing hand-engineered temporal features. An LSTM is suitable for this hypothesis because it can represent ordered within-patient observations and their short-term context.

## Dataset and split

The input is `data/raw/sepsis/Dataset.csv`. The existing patient-level train/test split is reused. A validation split is then made from training patients only. No patient can occur in more than one of training, validation, or test. The untouched test patients are used only for final evaluation.

The label is the existing `SepsisLabel` on the current observation. This preserves comparability with the current experiments. It is an experimental formulation, not a strict future-horizon prediction task; a separately specified horizon should be evaluated later if prospective prediction is required.

## Features and sequences

The initial feature set is:

`HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `Resp`, `WBC`, and `Lactate`.

`Patient_ID`, `SepsisLabel`, and time identifiers are not model inputs. The selected variables match the existing temporal experiment and are available clinical observations. Additional variables are intentionally deferred until their missingness, timing, and leakage properties are reviewed.

Rows are sorted by `Patient_ID` and `Hour`. For every current row, a window contains that row and up to the preceding 11 observations from the same patient. Windows are left-padded to length 12. Padding is zero after transformation and is ignored by a Keras `Masking` layer; no clinical values are invented. No future row or future label is used.

## Preprocessing and imbalance

Median imputation and `StandardScaler` are fit only on training-patient observations. The fitted transformer is then applied unchanged to validation and test observations. Class weights are calculated from training sequence labels only and passed to weighted binary cross entropy. SMOTE and synthetic oversampling are not used.

## Model and training

The default model is:

`Input -> Masking -> LSTM(64) -> Dropout(0.3) -> Dense(32, relu) -> Dense(1, sigmoid)`

The defaults are configurable in `LSTMConfig`: sequence length 12, batch size 256, learning rate 0.001, and a maximum of 25 epochs. Early stopping and best-checkpoint monitoring use validation PR-AUC. Random seeds are recorded and set where supported.

## Evaluation

The test report includes accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and a confusion matrix. Recall, precision, F1, PR-AUC, and ROC-AUC are the important metrics because the positive class is rare; accuracy is not sufficient evidence of useful performance.

Validation probabilities are evaluated at thresholds 0.10 through 0.60. The default operating point is selected by validation F1, with recall and precision as tie-breakers. This is an engineering operating point, not a clinically optimal threshold. The selected threshold is then applied once to the untouched test probabilities.

The comparison table reads the existing baseline Logistic Regression and temporal Logistic Regression metrics and adds the LSTM test metrics. The experiment reports the winner for recall, F1, PR-AUC, and ROC-AUC rather than assuming the LSTM is superior.

## Artifacts

Running `python -m src.sepsis.lstm_model` writes only to `models/sepsis_lstm/`:

- `sepsis_lstm.keras` and `best_model.keras`
- `preprocessor.joblib` and `inference_metadata.joblib`
- `metadata.json`, `split_summary.json`, `metrics.json`, and `training_history.json`
- `threshold_metrics.csv` and `model_comparison.csv`
- `plots/loss_curve.png`, `plots/pr_auc_curve.png`, `plots/roc_curve.png`, `plots/precision_recall_curve.png`, `plots/confusion_matrix.png`, and `plots/threshold_comparison.png`

The existing `models/temporal/` and `models/aki/` artifacts are not overwritten.

## Limitations

This is an academic research prototype. The current-label formulation does not establish a prospective clinical forecast, and the threshold is not clinically validated. Performance on this dataset does not establish real-world clinical effectiveness, safety, or clinical validity. The LSTM must not be described as a medical diagnosis system.