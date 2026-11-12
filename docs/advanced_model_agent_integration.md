# Advanced Model Agent Integration

This phase connects the completed experimental models to the existing disease detection agents. It does not retrain models, change training data, replace baseline artifacts, or implement orchestration, trend analysis, explainability, backend, or frontend components.

## Sepsis LSTM

`SepsisDetectionAgent` uses the saved LSTM by default from `models/sepsis_lstm/`. It loads `sepsis_lstm.keras`, `preprocessor.joblib`, and `inference_metadata.joblib`. The saved preprocessing transformer is applied as-is and is never refit during inference.

The input must contain one consistent `Patient_ID`, chronological `Hour` values, and the eight trained features: `HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `Resp`, `WBC`, and `Lactate`. At least 12 real observations are required. The agent sorts by hour and uses the latest 12 observations. It does not invent padding or synthetic clinical history. Fewer than 12 observations produce a clear validation error.

The saved threshold is applied to the LSTM probability. The result identifies `disease`, `model`, `probability`, `threshold`, `prediction`, `risk_level`, and `number_of_observations_used`. `model="baseline"` remains available for explicit use of the existing temporal Logistic Regression artifact.

## AKI XGBoost

`AKIDetectionAgent` uses the saved XGBoost model by default from `models/aki_xgboost/`. It loads `xgboost_model.json` as an XGBoost `Booster`, `preprocessor.joblib`, and `metadata.json`. The saved 60-feature strict formulation is used exactly as recorded in metadata.

Inputs must provide all 60 required features and a `patient_reference`. The agent rejects `AKI_TARGET`, `aki_stage`, `creat`, `CREATININE_min`, `CREATININE_max`, `uo_rt_6hr`, `uo_rt_12hr`, and `uo_rt_24hr`, as well as other excluded fields recorded in metadata. It applies the saved preprocessing and XGBoost inference without retraining. The saved operating threshold is then applied and the structured result identifies `disease`, `model`, `probability`, `threshold`, `prediction`, and `risk_level`.

`model="baseline"` remains available for explicit use of the existing AKI baseline pipeline.

## Model and agent responsibilities

The ML model learns patterns during training and produces probabilities or predictions. The detection agent validates incoming data, prepares it according to the saved training contract, invokes the frozen model, applies the configured threshold, maps the output to a prototype risk label, and returns structured information. These are model-backed software agents, not LLM agents.

## Demonstrations and limitations

The demo scripts use clearly labelled synthetic/test data. Their outputs are software integration checks, not real patient results. Both integrations are academic prototypes. Returned probabilities and risk labels are not medical diagnoses and are not clinically validated. Dataset performance does not establish real-world effectiveness, safety, or generalization. Baseline model selection remains available for research comparison and has not been deleted or overwritten.
