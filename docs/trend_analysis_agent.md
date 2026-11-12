# Trend Analysis Agent v1

## Purpose

`TrendAnalysisAgent` analyzes repeated model assessment results for one patient and one disease. It summarizes whether the model probability is moving upward, downward, or remaining approximately unchanged. It does not make a new ML prediction, retrain a model, diagnose a patient, recommend treatment, or assign final priority.

## Inputs

The agent accepts a non-empty sequence of `PatientAssessmentResult` objects or compatible dictionaries. Each usable record must provide `patient_id`, `disease`, and a numeric `probability` between 0 and 1. `assessment_time` is optional:

- When all records have assessment times, records are safely sorted by that field.
- When all records lack assessment times, the supplied sequence order is treated as the caller-provided history order.
- Mixing timed and untimed records is rejected because their ordering is ambiguous.

All records must belong to the same patient and the same supported disease: `Sepsis` or `AKI`. Sepsis and AKI histories are never combined into one trend.

## Output

The result dictionary contains:

- `patient_id`
- `disease`
- `trend`
- `first_probability`
- `latest_probability`
- `probability_change`
- `observation_count`
- `assessment_time_start`
- `assessment_time_end`
- `status`
- `message`

With fewer than two usable assessments, the agent returns `trend="INSUFFICIENT_DATA"` and does not fabricate a trend.

## Trend logic

The default engineering threshold is `0.05` absolute probability points:

- Change greater than `+0.05`: `WORSENING`
- Change less than `-0.05`: `IMPROVING`
- Change from `-0.05` through `+0.05`: `STABLE`

The calculation compares the earliest ordered probability with the latest ordered probability. Intermediate records are retained in the observation count and provide history context, but the v1 rule remains intentionally transparent and simple.

These thresholds are prototype software rules. They are not clinically validated deterioration thresholds and do not establish medical meaning.

## Example

A Sepsis history for `P001` changing from probability `0.25` to `0.72` produces `WORSENING`. An AKI history for `P002` changing from `0.78` to `0.31` produces `IMPROVING`.

## Limitations

The agent consumes model outputs; it does not rerun LSTM or XGBoost inference. It does not account for calibration, missing clinical context, treatment, sampling frequency, or disease-specific clinical interpretation. Its result is a software/data summary for later system components, not a clinical diagnosis, treatment recommendation, or validated decision rule.
