import pandas as pd
from pathlib import Path

raw = Path("data/raw/clinical_sample.csv")
out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(raw, parse_dates=["diagnosis_date"])

# Minimal cleanliness: drop obvious bad rows and enforce types
df = df.dropna(subset=["patient_id","diagnosis_date","age_at_diagnosis"])
df["event_timestamp"] = df["diagnosis_date"]

df.to_parquet(out_dir / "clinical_clean.parquet", index=False)
print("Wrote", out_dir / "clinical_clean.parquet")