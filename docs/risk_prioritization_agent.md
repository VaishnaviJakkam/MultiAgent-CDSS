# Risk Prioritization Agent v1

## Purpose

`RiskPrioritizationAgent` combines existing disease assessment probabilities with available disease trends to produce a prototype patient priority level. It does not run Sepsis LSTM or AKI XGBoost inference, retrain models, diagnose a patient, recommend treatment, or replace the disease detection agents.

## Inputs

The agent accepts one or more assessment result objects or compatible dictionaries for one patient. Supported diseases are `Sepsis` and `AKI`. Each assessment must include `patient_id`, `disease`, and a probability between 0 and 1. Duplicate disease assessments are rejected explicitly rather than silently ignored.

Trend results are optional. When supplied, each trend must match the assessment patient and disease and use a supported trend value: `WORSENING`, `IMPROVING`, `STABLE`, or `INSUFFICIENT_DATA`. Missing trends are represented explicitly in the output.

## Output

The result includes:

- `patient_id`
- `priority_level`
- `diseases_assessed`
- `highest_probability`
- `highest_risk_disease`
- `worsening_diseases`
- `available_trends`
- `reason`
- `status`
- `message`

`priority_level` is a patient-level prioritization output. It must not be confused with a disease-level `risk_level` from Sepsis or AKI assessment results.

## Prototype priority logic

The named constants are:

- `HIGH_PROBABILITY_THRESHOLD = 0.70`
- `MODERATE_PROBABILITY_THRESHOLD = 0.40`
- `WORSENING_HIGH_PROBABILITY_THRESHOLD = 0.50`

The rules are evaluated as follows:

- `HIGH` if any disease probability is at least 0.70.
- `HIGH` if any disease probability is at least 0.50 and its trend is `WORSENING`.
- `HIGH` if both diseases have probability at least 0.40 and at least one has a worsening trend.
- `MEDIUM` if any probability is at least 0.40, or a worsening trend exists without a high-priority condition, or both diseases have moderate risk.
- `LOW` otherwise.

Each result includes a plain-language reason for the selected priority. No treatment or clinical recommendation is generated.

## Distinguishing responsibilities

- Disease prediction: Sepsis and AKI agents invoke frozen ML models and return disease probabilities.
- Trend analysis: `TrendAnalysisAgent` compares repeated probabilities for one patient and disease.
- Patient prioritization: this agent combines available disease probabilities and trends into one software priority level.

## Limitations

The prioritization rules implemented here are prototype engineering rules and are not clinically validated or intended to replace professional clinical judgment. The thresholds are not clinical guidelines, and the output is not a diagnosis or treatment recommendation. The agent does not account for calibration, missing clinical context, time intervals, comorbidities, or clinical workflow.
