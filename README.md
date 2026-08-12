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
