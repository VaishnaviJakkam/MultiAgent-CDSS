# Patient Database Layer

## Purpose

The database layer provides longitudinal application storage for admitted patients. It preserves patient observations, disease assessments, trend results, and prioritization results as separate records so later application components can retrieve patient-specific history without mixing admissions.

The patient's stored history is separate from the datasets used to train the Sepsis and AKI machine-learning models.

## MongoDB usage

Production persistence uses MongoDB through `pymongo`. The connection URI is read from the `MONGODB_URI` environment variable, and the optional database name is read from `MONGODB_DATABASE` or defaults to `multiagent_cdss`. Credentials and connection strings are never hard-coded.

The connection helper is `src/database/connection.py`. It returns a database handle; FastAPI integration is intentionally not included in this step.

## Data model

The repository uses separate collections:

- `patients`: stable `patient_id`, demographics, status, and lifecycle timestamps.
- `admissions`: `patient_id` plus stable `admission_id`, admission time, status, and timestamps.
- `observations`: one immutable clinical observation per submission, with `observation_id`, patient/admission references, observation time, supplied clinical parameters, source, and creation time.
- `assessments`: one model execution result per submission, with patient/admission references, assessment time, and Sepsis/AKI result snapshots.
- `trends`: disease-specific trend records with probabilities, change, count, trend, and time.
- `prioritizations`: patient-level priority records with highest-risk disease, probability, worsening diseases, reason, and time.

A patient may have multiple admissions. The pair `patient_id + admission_id` scopes every longitudinal record. A new observation or assessment is inserted as a new document; previous records are not overwritten.

## Repository operations

`PatientRepository` provides:

- `create_patient()` and `get_patient()`
- `create_admission()` and `get_admission()`
- `add_observation()` and `get_patient_observations()`
- `add_assessment()` and `get_patient_assessments()`
- `add_trend()` and `get_patient_trends()`
- `add_prioritization()` and `get_latest_prioritization()`
- `get_patient_history()`

The complete history method verifies the admission scope and returns chronologically ordered observations and assessments, trend records, and the latest prioritization. It never combines records from another patient or admission.

## Validation boundary

The repository validates non-empty identifiers, datetime timestamps, existing patient/admission references, and admission scope. It does not apply clinical parameter rules. Disease agents remain responsible for model input validation, while later application services can add request-specific validation.

Tests use `InMemoryDatabase`, a small Mongo-like adapter in the repository module. This keeps focused tests deterministic and avoids requiring a live MongoDB server. Production code remains compatible with a MongoDB database handle.

## Future Trend Analysis support

Each assessment is retained with its assessment time and disease result. Future Trend Analysis can retrieve all assessments for one `patient_id + admission_id`, order them, and pass one disease's probability history to the existing Trend Analysis Agent. Trends and prioritizations are also stored separately, preserving an auditable progression of derived outputs.

This layer is storage only. It does not create REST endpoints, run disease models, modify model artifacts, call Trend Analysis or Risk Prioritization, or provide clinical advice.
