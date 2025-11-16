import pandas as pd
from pathlib import Path

inp = Path("data/processed/clinical_clean.parquet")
out = Path("data/processed/features.parquet")

df = pd.read_parquet(inp)

# Simple derived features (toy)
df["age_bin"] = pd.cut(df["age_at_diagnosis"], bins=[0,40,50,60,120], labels=["<=40","41-50","51-60","60+"])
df["is_erpos"] = (df["er_status"].str.lower()=="positive").astype(int)
df["is_prpos"] = (df["pr_status"].str.lower()=="positive").astype(int)
df["is_her2pos"] = (df["her2_status"].str.lower()=="positive").astype(int)

# Select columns for the feature table
feat = df[[
    "patient_id","event_timestamp","age_at_diagnosis","age_bin","ajcc_stage",
    "is_erpos","is_prpos","is_her2pos","overall_survival_days","event"
]].copy()

feat.to_parquet(out, index=False)
print("Wrote", out)