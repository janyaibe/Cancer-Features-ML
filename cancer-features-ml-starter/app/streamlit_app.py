import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

st.set_page_config(page_title="Cancer Features Demo", layout="wide")

st.title("Cancer Features Demo (educational only)")
st.caption("Toy data and model — not clinical advice.")

data_path = Path("data/processed/features.parquet")
model_path = Path("models/baseline_model.pkl")

if not data_path.exists():
    st.error("Run `make features` first to build features.")
else:
    df = pd.read_parquet(data_path)
    st.subheader("Feature snapshot")
    st.dataframe(df.head(10))

    if model_path.exists():
        clf = joblib.load(model_path)
        st.subheader("Try a prediction")
        pid = st.selectbox("Choose a patient_id", df["patient_id"].unique())
        row = df[df["patient_id"]==pid].iloc[0]
        X = pd.DataFrame([{
            "age_at_diagnosis": row["age_at_diagnosis"],
            "ajcc_stage": row["ajcc_stage"],
            "is_erpos": row["is_erpos"],
            "is_prpos": row["is_prpos"],
            "is_her2pos": row["is_her2pos"],
        }])
        proba = clf.predict_proba(X)[0,1]
        st.metric("Predicted risk score (toy)", f"{proba:.2f}")
    else:
        st.info("Train a model with `make train` to enable predictions.")