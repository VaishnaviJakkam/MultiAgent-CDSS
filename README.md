# MultiAgent-CDSS

This repo contains the Sepsis clinical decision support project with patient-level preprocessing, model training, threshold analysis, and temporal feature experiments.

## Project structure

- `src/` — Python source code for preprocessing, splitting, training, and validation
- `notebooks/` — exploratory notebooks
- `requirements.txt` — project dependencies
- `inspect_output.txt` — dataset inspection notes
- `data/` — local dataset folder, not pushed to GitHub
- `models/` — local training outputs, not pushed to GitHub

## Important note

The large `data/` and `models/` folders are intentionally excluded from GitHub. Keep them on your local machine so the project can run correctly.

## Step-by-step: how to run this project

### Step 1: Clone the repository

```bash
git clone https://github.com/VaishnaviJakkam/MultiAgent-CDSS.git
cd MultiAgent-CDSS
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate it:

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
venv\Scripts\activate.bat
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Make sure the local data is present

The project expects the dataset in:

```text
data/raw/sepsis/Dataset.csv
```

If that file is missing locally, copy it from your local project directory or the original dataset source before running the training code.

### Step 5: Run the Sepsis training pipeline

From the project root:

```bash
python -m src.sepsis.train
```

This trains the baseline models and writes results into the local `models/` directory.

### Step 6: Run the test suite

```bash
pytest -q
```

Or run a specific file:

```bash
pytest -q src/sepsis/test_split.py
pytest -q src/sepsis/test_preprocessing.py
pytest -q src/sepsis/test_training.py
pytest -q src/sepsis/test_temporal_features.py
```

### Step 7: Run the temporal feature experiment

```bash
python -m src.sepsis.temporal_features
```

This experiment adds patient-history temporal features while keeping the patient-level split integrity intact.

## Repo behavior

- Source code and README are tracked in Git.
- `data/` and `models/` stay local and are not pushed.
- The repo is meant to be used with a local dataset and local model outputs.

## Troubleshooting

- If `python -m src.sepsis.train` fails, verify you are in the project root and the virtual environment is active.
- If imports fail, reinstall dependencies with `pip install -r requirements.txt`.
- If the data is missing, check that `data/raw/sepsis/Dataset.csv` exists locally.

## Usage note

This project is a research and prototype decision-support workflow. It is not a clinical decision system and should be reviewed appropriately before real-world use.

## Sepsis Detection Agent

Purpose: Prototype sepsis risk estimation agent that produces a probability-based decision-support output. It is NOT a clinical diagnosis or treatment recommendation tool.

Input: A patient history (list of structured observations) where each observation is a mapping with `Patient_ID`, `Hour` (or another time column), and clinical features such as `HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `Resp`, `WBC`, `Lactate`, etc. Example:

```python
[
	{"Patient_ID": "P1", "Hour": 0, "HR": 80, "O2Sat": 98, ...},
	{"Patient_ID": "P1", "Hour": 1, "HR": 82, "O2Sat": 97, ...},
]
```

Processing: The agent reuses the project's `src/sepsis/temporal_features.py` functions to generate lag, change, percentage-change, and rolling temporal features using only information available up to each observation. The saved temporal pipeline `models/temporal/sepsis_temporal_pipeline.pkl` is loaded and used to produce a positive-class probability via `predict_proba()`.

ML model used: The frozen temporal logistic regression pipeline saved at `models/temporal/sepsis_temporal_pipeline.pkl` (prototype, not clinically validated).

Output: A structured dictionary consumable by other agents, for example:

```json
{
	"agent": "SepsisDetectionAgent",
	"patient_id": "P1001",
	"disease": "Sepsis",
	"prediction": 1,
	"probability": 0.572,
	"threshold": 0.5,
	"risk_level": "ELEVATED",
	"status": "success"
}
```

Temporal feature handling: The agent calls `add_temporal_features()` and `sort_patient_observations()` from `src/sepsis/temporal_features.py` so the temporal feature logic is identical to the training experiment.

Limitations: This is an academic prototype. The model is not clinically validated. The agent does not store data, recommend treatments, or perform explanations. Use only with synthetic or appropriately consented data for testing.

## AKI Detection Agent

Purpose: Prototype AKI risk estimation agent for cross-sectional inpatient data. This agent is intentionally strict about feature inputs and excludes creatinine and urine-output variables that could directly leak the AKI staging target.

Input: A single patient report dictionary containing `patient_reference` and the exact feature set saved in `models/aki/aki_metadata.json` under `feature_names`. Example:

```python
{
    "patient_reference": "P1001",
    "ANIONGAP_min": 8.0,
    "ANIONGAP_max": 12.0,
    "ALBUMIN_min": 3.8,
    "ALBUMIN_max": 4.2,
    # ... all other required AKI features ...
}
```

Processing: The agent loads `models/aki/aki_pipeline.pkl` and uses the saved preprocessing/model pipeline to compute a positive-class probability. It validates the request and rejects payloads that include excluded leakage fields such as `aki_stage`, `creat`, `CREATININE_min`, `CREATININE_max`, `uo_rt_6hr`, `uo_rt_12hr`, or `uo_rt_24hr`.

Usage:

```bash
python src/agents/demo_aki_agent.py
```

Output: The agent returns a structured result with fields such as `prediction`, `probability`, `threshold`, `risk_level`, and `status`.

Limitations: This is a prototype only. It is not a clinical diagnosis or treatment tool and should be used only for local testing with appropriately consented data.

