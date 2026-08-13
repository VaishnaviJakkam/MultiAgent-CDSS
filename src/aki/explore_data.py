from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import RAW_AKI_DATA_PATH

KEYWORDS = ["aki", "stage", "creat", "id", "patient", "diagnosis", "label"]
TIME_CANDIDATES = ["time", "date", "hour", "day", "admit", "icd", "icu", "los", "elapsed"]


def load_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def find_keyword_columns(columns: list[str], keywords: list[str]) -> list[str]:
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keywords) + r")\b", flags=re.IGNORECASE)
    return [col for col in columns if pattern.search(col)]


def candidate_distribution(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        print(f"\nCandidate column: {col}")
        series = df[col]
        print(f"  dtype={series.dtype}, missing={series.isna().mean():.4f}, unique={series.nunique(dropna=False)}")
        if series.dtype.kind in "bifc" and series.nunique(dropna=False) <= 20:
            print(series.value_counts(dropna=False).sort_index().to_string())
        elif series.dtype == object and series.nunique(dropna=False) <= 20:
            print(series.value_counts(dropna=False).to_string())


def inspect_aki_workbook(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"AKI dataset not found at {path}")

    print("AKI Dataset Inspection")
    print("=====================")
    print(f"Path: {path}")
    print(f"File type: {path.suffix}")

    df = load_data(path)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    if path.suffix.lower() in {".xls", ".xlsx"}:
        xl = pd.ExcelFile(path)
        print(f"Sheet names: {xl.sheet_names}")
        print(f"Number of sheets: {len(xl.sheet_names)}")
    else:
        print("Sheet names: ['Raw CSV']")
        print("Number of sheets: 1")

    print("\nColumn overview")
    print("---------------")
    for col in df.columns:
        missing_pct = 100.0 * df[col].isna().mean()
        unique_count = df[col].nunique(dropna=False)
        print(f"{col}: dtype={df[col].dtype}, missing={missing_pct:.2f}%, unique={unique_count}")

    print("\nSample records")
    print("--------------")
    print(df.head(10).to_string(index=False))

    print("\nMissing value summary")
    print("---------------------")
    missing = df.isna().mean().sort_values(ascending=False)
    print(missing[missing > 0].to_string())

    print("\nDuplicate row summary")
    print("---------------------")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\nPotential target and identifier candidates")
    print("------------------------------------------")
    target_candidates = find_keyword_columns(df.columns.tolist(), KEYWORDS)
    print(f"Columns matching target/identifier keywords: {target_candidates}")
    candidate_distribution(df, target_candidates)

    print("\nExplicit patient identifier candidates")
    print("------------------------------------")
    id_candidates = [
        col for col in df.columns if col.lower() in {"patient_id", "patientid", "id", "subjectid", "subject_id", "record_id"}
    ]
    if id_candidates:
        print(id_candidates)
    else:
        print("No patient identifier was found.")

    print("\nTemporal field candidates")
    print("--------------------------")
    temporal_candidates = [col for col in df.columns if any(t in col.lower() for t in TIME_CANDIDATES)]
    print(temporal_candidates if temporal_candidates else "No explicit temporal columns were found.")

    print("\nDataset structure assessment")
    print("----------------------------")
    if not id_candidates and not temporal_candidates:
        print("No explicit patient ID or temporal field detected. The dataset appears cross-sectional or aggregated with one row per record.")
    elif id_candidates and temporal_candidates:
        print("The dataset contains both an identifier and a temporal field and may be longitudinal.")
    elif id_candidates:
        print("The dataset contains a patient identifier but no clear temporal field.")
    else:
        print("No explicit patient ID found; temporal structure cannot be confirmed.")

    print("\nData type summary")
    print("-------------------")
    print(df.dtypes.value_counts().to_string())

    print("\nEnd of inspection report")


if __name__ == "__main__":
    inspect_aki_workbook(RAW_AKI_DATA_PATH)
