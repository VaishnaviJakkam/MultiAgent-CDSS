from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "sepsis" / "Dataset.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)
rows = len(df)
print('shape', df.shape)
print('columns', list(df.columns))
print()
print('column,type,missing%,unique')
for c in df.columns:
    missing = df[c].isna().sum()
    uniq = df[c].nunique(dropna=True)
    print(f'{c},{df[c].dtype},{missing/rows:.6f},{uniq}')
print()
print('low-cardinality columns')
for c in df.columns:
    if df[c].nunique(dropna=True) <= 20:
        values = sorted(df[c].dropna().unique())
        print(c, values)
