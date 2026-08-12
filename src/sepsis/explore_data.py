from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_SEPSIS_DIR = PROJECT_ROOT / "data" / "raw" / "sepsis"


def summarize_dataframe(df: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"Shape: {df.shape}")
    print("Columns:")
    print("  ".join(df.columns.astype(str).tolist()))
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values by column:")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0])
    print("\nSample unique values for low-cardinality columns:")
    for col in df.columns:
        if 1 < df[col].nunique(dropna=False) <= 20:
            values = df[col].dropna().unique()
            print(f"  {col}: {sorted(values)[:20]}")
    if "SepsisLabel" in df.columns:
        print("\nTarget distribution (SepsisLabel):")
        print(df["SepsisLabel"].value_counts(dropna=False))


def inspect_dataset_csv(dataset_path: Path) -> pd.DataFrame:
    print(f"Inspecting combined dataset file: {dataset_path}")
    df = pd.read_csv(dataset_path, low_memory=False)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=[col for col in df.columns if col.startswith("Unnamed")])
    if "" in df.columns:
        df = df.rename(columns={"": "index"})
    summarize_dataframe(df, "Dataset.csv")
    if "Patient_ID" in df.columns:
        print(f"Unique patients: {df['Patient_ID'].nunique()}\n")
        obs_per_patient = df["Patient_ID"].value_counts()
        print("Observations per patient - sample distribution:")
        print(obs_per_patient.describe())
    return df


def inspect_psv_directories(root: Path) -> None:
    print(f"\nInspecting PSV directories under {root}")
    psv_files = list(root.rglob("*.psv"))
    print(f"Found {len(psv_files)} .psv files in {root}")
    if not psv_files:
        return
    sample_file = psv_files[0]
    print(f"Reading sample PSV file: {sample_file}")
    sample_df = pd.read_csv(sample_file, sep="|", na_values=["NaN", ""], low_memory=False)
    summarize_dataframe(sample_df, f"Sample file {sample_file.name}")
    if "SepsisLabel" in sample_df.columns:
        print("\nSample file target distribution:")
        print(sample_df["SepsisLabel"].value_counts(dropna=False))
    print("\nSample file patient identifier:")
    print(f"  Derived from filename: {sample_file.stem}")


def inspect_raw_data() -> None:
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw sepsis data directory: {RAW_SEPSIS_DIR}")
    if not RAW_SEPSIS_DIR.exists():
        raise FileNotFoundError(f"Sepsis raw directory not found: {RAW_SEPSIS_DIR}")

    top_level = sorted([p.name for p in RAW_SEPSIS_DIR.iterdir()])
    print(f"\nTop-level contents: {top_level}")

    dataset_path = RAW_SEPSIS_DIR / "Dataset.csv"
    if dataset_path.exists():
        try:
            inspect_dataset_csv(dataset_path)
        except Exception as exc:
            print(f"Error reading Dataset.csv: {exc}")
    else:
        print("Dataset.csv not found.")

    for subdir_name in ["training_setA", "training_setB"]:
        subdir = RAW_SEPSIS_DIR / subdir_name
        if subdir.exists():
            if subdir_name == "training_setA":
                inspect_psv_directories(subdir)
            else:
                print(f"\nFound {subdir_name} directory at {subdir}. It likely contains additional patient-level PSV files.")
                count = sum(1 for _ in subdir.rglob("*.psv"))
                print(f"  {subdir_name} PSV file count: {count}")
        else:
            print(f"{subdir_name} not found.")


if __name__ == "__main__":
    inspect_raw_data()
