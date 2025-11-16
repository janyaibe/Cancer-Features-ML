import sys
import pandas as pd
from pathlib import Path

p = Path("data/processed/features.parquet")
df = pd.read_parquet(p)

errors = []

# Schema checks
required = ["patient_id","event_timestamp","age_at_diagnosis","overall_survival_days","event"]
missing = [c for c in required if c not in df.columns]
if missing:
    errors.append(f"Missing required columns: {missing}")

# Basic ranges
if (df["age_at_diagnosis"] < 0).any():
    errors.append("age_at_diagnosis has negative values")

if (df["overall_survival_days"] < 0).any():
    errors.append("overall_survival_days has negative values")

# Null ratios
for c in required:
    null_rate = df[c].isna().mean()
    if null_rate > 0.01:
        errors.append(f"High null rate in {c}: {null_rate:.2%}")

if errors:
    print("\n".join(errors))
    sys.exit(1)
else:
    print("Basic data checks passed.")