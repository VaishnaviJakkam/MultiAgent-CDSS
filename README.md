# MultiAgent-CDSS

## What is this repo?

MultiAgent-CDSS is a research prototype for clinical decision support using machine-learning agents. It provides Sepsis and AKI risk estimation, along with agents for trend analysis and risk prioritization.

## Repository structure

```text
MultiAgent-CDSS/
|-- data/                 Local datasets (not pushed to GitHub)
|-- models/               Local trained models and results (not pushed to GitHub)
|-- docs/                 Project and experiment documentation
|-- src/
|   |-- agents/           Sepsis, AKI, trend, and risk agents
|   |-- aki/              AKI preprocessing and model code
|   |-- database/         MongoDB connection and repository code
|   |-- sepsis/           Sepsis preprocessing and model code
|-- tests/                Project tests
|-- requirements.txt      Python dependencies
|-- documentation.md      Detailed project documentation
```

## Local setup and run

### 1. Clone the repository

```bash
git clone https://github.com/VaishnaviJakkam/MultiAgent-CDSS.git
cd MultiAgent-CDSS
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Add local project data

Keep datasets and trained model files in the local `data/` and `models/` folders. The Sepsis dataset should be available at:

```text
data/raw/sepsis/Dataset.csv
```

### 5. Run the Sepsis pipeline

```bash
python -m src.sepsis.train
```

### 6. Run the agent demos

Synthetic demos:

```bash
python -m src.agents.demo_trend_analysis_agent
python -m src.agents.demo_risk_prioritization_agent
```

Model-based demos, after the required local models have been generated:

```bash
python -m src.agents.demo_sepsis_agent
python -m src.agents.demo_aki_agent
```

### 7. Run the tests

```bash
pytest -q
```
