# MultiAgent-CDSS

MultiAgent-CDSS is a research prototype for combining disease-specific machine-learning agents into a clinical decision-support workflow. The current implementation covers Sepsis and AKI risk estimation, shared result validation, trend analysis, risk prioritization, and an optional MongoDB persistence layer.

This project is not a clinical diagnosis, treatment recommendation, or production-ready clinical system.

## Current components

- `src/sepsis/` — Sepsis preprocessing, patient-level splitting, training, temporal features, and LSTM experiments.
- `src/aki/` — AKI preprocessing, leakage-aware splitting, training, and XGBoost experiments.
- `src/agents/sepsis_agent.py` — Sepsis detection agent.
- `src/agents/aki_agent.py` — AKI detection agent.
- `src/agents/result_schema.py` — shared `PatientAssessmentResult` contract.
- `src/agents/trend_analysis_agent.py` — compares assessments over time.
- `src/agents/risk_prioritization_agent.py` — combines disease risk and trends into a priority level.
- `src/database/` — MongoDB connection, initialization, models, and repositories.
- `tests/` and `src/**/test_*.py` — unit and integration tests.
- `docs/` and `documentation.md` — detailed implementation notes and experiment documentation.

The agents are lightweight Python components. They validate structured input, call a saved model or receive a normalized assessment, and return machine-readable results for downstream agents.

## Repository data policy

The following remain local and are intentionally excluded from GitHub:

- `data/` — raw and processed datasets.
- `models/` — trained models, preprocessors, metrics, and experiment outputs.
- `.env` — local configuration and credentials.
- `venv/` and Python/test caches.

Never commit patient data, database credentials, API keys, or trained artifacts containing sensitive information. Use synthetic or appropriately consented data for local testing.

## Setup

Clone the project and create an isolated environment:

```bash
git clone https://github.com/VaishnaviJakkam/MultiAgent-CDSS.git
cd MultiAgent-CDSS
python -m venv venv
```

Activate the environment on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Local datasets and model artifacts

Training and model-backed agent demos require the corresponding local files. For Sepsis, the raw dataset is expected at:

```text
data/raw/sepsis/Dataset.csv
```

AKI training expects the local AKI dataset described in the project documentation. Run the training scripts to generate the required files under `models/`. Do not copy datasets or model artifacts into the Git repository.

## Running the project

Run Sepsis training from the repository root:

```bash
python -m src.sepsis.train
```

Run the temporal feature experiment:

```bash
python -m src.sepsis.temporal_features
```

Run the synthetic agent demos, which do not require patient data:

```bash
python -m src.agents.demo_trend_analysis_agent
python -m src.agents.demo_risk_prioritization_agent
```

The Sepsis and AKI detection demos require their corresponding locally generated model artifacts:

```bash
python -m src.agents.demo_sepsis_agent
python -m src.agents.demo_aki_agent
```

## MongoDB configuration

The database layer reads configuration from a local `.env` file. Create it in the repository root without committing it:

```text
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=multiagent_cdss
```

MongoDB is only needed for database-backed workflows and tests. The model experiments and synthetic agent demos do not require it.

## Testing

Run the complete suite:

```bash
pytest -q
```

Useful focused checks include:

```bash
pytest -q tests
pytest -q src/agents/test_result_schema.py
pytest -q src/agents/test_trend_analysis_agent.py
pytest -q src/agents/test_risk_prioritization_agent.py
pytest -q src/sepsis/test_temporal_features.py
```

## Documentation

See [documentation.md](documentation.md) for the detailed architecture, data flow, model experiments, agent contracts, and implementation status. Experiment-specific notes are available in [`docs/`](docs/).

## Limitations

This is an academic prototype. It does not provide clinical validation, explainable predictions, automated EHR/report ingestion, a production API, authentication, or a frontend dashboard. Outputs must be reviewed by qualified professionals before any real-world use.
