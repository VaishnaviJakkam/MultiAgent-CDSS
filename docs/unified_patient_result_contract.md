# Unified Patient Assessment Result Contract

The disease agents now return a common software/data contract for successful model assessments. This gives future Trend Analysis, Risk Prioritization, Explainability, and orchestration components a stable representation without coupling them to disease-specific model internals.

## Contract fields

`PatientAssessmentResult` is defined in `src/agents/result_schema.py` and serializes with `to_dict()`.

- `patient_id`: patient identifier or supplied patient reference.
- `disease`: disease name, currently `Sepsis` or `AKI`.
- `model`: model used for the result, such as `LSTM`, `XGBoost`, or an explicit baseline model.
- `prediction`: binary model decision, `0` or `1`.
- `probability`: model probability for the positive class.
- `threshold`: operating threshold used for the binary decision.
- `risk_level`: existing prototype label, `LOWER` or `ELEVATED`.
- `observation_count`: number of observations represented by the assessment.
- `assessment_time`: safely derived input time when available; Sepsis uses the latest real `Hour`, while cross-sectional AKI has `None`.
- `status`: successful model result status, currently `success`.
- `message`: software-result context. It explicitly states that the output is not a medical diagnosis.

The schema validates binary predictions, probability and threshold ranges, and non-negative observation counts. Existing validation errors remain explicit: Sepsis continues to raise validation exceptions, and AKI continues to return structured error dictionaries.

## Sepsis example

```json
{
  "patient_id": "SYNTHETIC-TEST-001",
  "disease": "Sepsis",
  "model": "LSTM",
  "prediction": 1,
  "probability": 0.89,
  "threshold": 0.6,
  "risk_level": "ELEVATED",
  "observation_count": 12,
  "assessment_time": 11,
  "status": "success",
  "message": "Academic prototype output; not a medical diagnosis."
}
```

Sepsis requires 12 real chronological observations. No history is invented or silently padded by the agent.

## AKI example

```json
{
  "patient_id": "SYNTHETIC-TEST-001",
  "disease": "AKI",
  "model": "XGBoost",
  "prediction": 0,
  "probability": 0.39,
  "threshold": 0.6,
  "risk_level": "LOWER",
  "observation_count": 1,
  "assessment_time": null,
  "status": "success",
  "message": "Academic prototype output; not a medical diagnosis."
}
```

AKI is cross-sectional, so one structured observation is represented and no temporal assessment time is inferred.

## Future consumers

Future components can consume the common fields without knowing whether the underlying model is an LSTM, XGBoost model, or baseline model. They may compare timestamps where present, aggregate disease outputs by `patient_id`, and preserve the original probability and threshold. Those future components must not reinterpret this contract as treatment advice or a clinically validated decision rule.

This is a software/data contract for an academic prototype, not a clinical rule and not a medical diagnosis.