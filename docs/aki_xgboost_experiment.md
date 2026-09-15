# Experimental AKI XGBoost Model

This experiment adds an isolated XGBoost model for the existing cross-sectional AKI dataset. It does not replace the existing Logistic Regression, Decision Tree, or Random Forest models, and it does not change the AKI Detection Agent.

## Dataset and target

The input is `data/raw/aki/Raw_aki_patient_data.csv`. The dataset has no explicit patient identifier or longitudinal time field, so the existing stratified observation-level split is reused. The target remains unchanged:

- `aki_stage == 0` becomes `AKI_TARGET = 0`.
- `aki_stage in {1, 2, 3}` becomes `AKI_TARGET = 1`.

## Leakage control and features

The strict feature set is obtained through the existing `get_strict_feature_columns` implementation. It excludes `aki_stage`, `creat`, `CREATININE_min`, `CREATININE_max`, `uo_rt_6hr`, `uo_rt_12hr`, and `uo_rt_24hr`, as well as the existing row index column. `Patient_ID`, `AKI_TARGET`, and any other forbidden target-derived columns are also rejected explicitly before training. The resulting raw feature count is 60.

## Preprocessing and split

The existing preprocessing convention is reused: numerical columns receive median imputation and standard scaling; categorical columns receive most-frequent imputation and one-hot encoding. The training observations are divided into a stratified train subset and validation subset. The preprocessor is fitted only on the train subset, then applied unchanged to validation and the untouched test set.

## XGBoost configuration

The lightweight starting configuration is 300 estimators, maximum depth 5, learning rate 0.05, row subsampling 0.8, column subsampling 0.8, random state 42, and all available CPU workers. The objective is `binary:logistic`, the evaluation metric is `aucpr`, and histogram tree construction is used. Validation PR-AUC controls early stopping. No synthetic oversampling is used.

Class imbalance is handled with `scale_pos_weight = negative_training_rows / positive_training_rows`, calculated from the train subset only. Validation and test labels are not used for this calculation.

## Threshold and metrics

Validation probabilities are evaluated at thresholds 0.20, 0.30, 0.40, 0.50, 0.60, and 0.70. The operating threshold is selected by validation F1, with recall and precision as tie-breakers. The selected threshold is applied once to test probabilities. This threshold is an experimental operating point and is not clinically optimal or clinically validated.

The untouched test set is evaluated with accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC. The comparison table includes all existing AKI models and XGBoost. The preferred model for this project's high-risk detection objective is assessed primarily using recall, F1, PR-AUC, and ROC-AUC rather than accuracy alone.

## Feature importance

The experiment saves XGBoost gain-based feature importance. Importance describes model association or usefulness within this fitted model; it does not establish causality, clinical importance, or medical significance.

## Completed experiment results

The run used 35,899 training rows, 8,975 validation rows, and 11,219 untouched test rows. Early stopping selected 135 boosting rounds from the 300-round maximum. The validation-selected operating threshold was 0.60. Test metrics at that threshold were accuracy 0.8093, precision 0.4312, recall 0.5014, and F1 0.4637; threshold-independent ROC-AUC was 0.7736 and PR-AUC was 0.4501.

Compared with the existing models, XGBoost had the highest F1, PR-AUC, and ROC-AUC. It did not improve recall over Logistic Regression (0.5014 versus 0.6455), and it did not improve precision over Random Forest (0.4312 versus 0.5271). Therefore, XGBoost is not automatically the preferred model for a recall-first objective, despite its stronger overall ranking on F1, PR-AUC, and ROC-AUC.

## Artifacts and limitations

All new outputs are written under `models/aki_xgboost/`. Existing `models/aki/` artifacts remain separate. This is an academic research prototype and is not a clinically validated diagnostic system. Dataset performance does not establish real-world clinical effectiveness, safety, or generalization.