from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sepsis" / "Dataset.csv"
TARGET = "SepsisLabel"
PATIENT_ID_COLUMN = "Patient_ID"
TIME_COLUMNS = ["Hour", "HospAdmTime", "ICULOS"]

# Clinical features available at each observation/timepoint
FEATURE_COLUMNS = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "EtCO2",
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
    "Age",
    "Gender",
]

NUMERICAL_FEATURES = [
    col for col in FEATURE_COLUMNS if col != "Gender"
]

CATEGORICAL_FEATURES = ["Gender"]

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
THRESHOLD_CANDIDATES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
DEFAULT_CLASSIFICATION_THRESHOLD = 0.50
PROTOTYPE_RISK_MARGIN = 0.15

# Columns that are not used as predictive features in the initial phase
# because they are identifiers, labels, or system/time metadata.
EXCLUDED_COLUMNS = ["Unnamed: 0", "Patient_ID", "SepsisLabel", "Unit1", "Unit2", "HospAdmTime"]
