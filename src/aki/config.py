from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_AKI_DATA_PATH = ROOT / "data" / "raw" / "aki" / "Raw_aki_patient_data.csv"

# Phase 2 AKI problem definition / inspection configuration.
# This dataset appears to be cross-sectional: no explicit patient ID or temporal history fields were found.
# Do NOT train any models in Phase 2, and do NOT use temporal patient history as part of the AKI definition.
TARGET_COLUMN = "aki_stage"
BINARY_TARGET_COLUMN = "AKI_TARGET"
BINARY_TARGET_MAPPING = {0: 0, 1: 1, 2: 1, 3: 1}
CROSS_SECTIONAL_ONLY = True

# Candidate columns that likely define or strongly relate to AKI stage.
# These should be reviewed carefully for leakage before including them in any predictive model.
POTENTIAL_TARGET_DEFINING_COLUMNS = [
    "creat",
    "CREATININE_min",
    "CREATININE_max",
    "uo_rt_6hr",
    "uo_rt_12hr",
    "uo_rt_24hr",
]

# Columns that are known to be labels, indexes, or otherwise not predictive features.
EXCLUDED_COLUMNS = ["Unnamed: 0"]

# Strict baseline feature exclusions for the first leakage-aware AKI model.
STRICT_BASELINE_EXCLUDED_COLUMNS = EXCLUDED_COLUMNS + [
    "aki_stage",
    "creat",
    "CREATININE_min",
    "CREATININE_max",
    "uo_rt_6hr",
    "uo_rt_12hr",
    "uo_rt_24hr",
]

# Clinical extended feature set may later include target-defining variables for comparison.
CLINICAL_EXTENDED_EXCLUDED_COLUMNS = EXCLUDED_COLUMNS + ["aki_stage"]

# Phase 3 preprocessing configuration.
TEST_SIZE = 0.20
RANDOM_STATE = 42
