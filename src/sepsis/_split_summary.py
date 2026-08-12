from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.sepsis.split_data import load_raw_dataset, patient_level_train_test_split, summarize_split

df = load_raw_dataset()
X_train, X_test, y_train, y_test, train_ids, test_ids = patient_level_train_test_split(df)
summary = summarize_split(df, train_ids, test_ids)
for k,v in summary.items():
    print(k, v)
