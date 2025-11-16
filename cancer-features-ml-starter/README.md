# Cancer Features → ML (Starter)

This repo is a small, reproducible scaffold that turns open cancer data into machine‑learning‑ready features with data contracts, drift checks, a baseline model tracked in MLflow, and a tiny Streamlit demo.

## What’s included
- Data contracts with a starter validation script
- Point‑in‑time friendly schema with event timestamps
- Feature definitions using Feast (local registry and online store)
- A minimal baseline model with MLflow tracking
- A Streamlit demo to explore features and run a toy prediction
- Make targets to run common workflows

## Quickstart
1) Create and activate a virtual environment, then install requirements.
2) `make data` to generate small processed data from `data/raw/clinical_sample.csv`.
3) `make features` to write a Parquet of features and refresh the Feast registry.
4) `make train` to train a baseline logistic model and log to MLflow.
5) `make app` to launch the Streamlit demo.

## Deliverables to show publicly
- `docs/quality/` data checks report (generated)
- Screenshot of Feast registry and feature definitions
- MLflow run page with dataset and params
- Short video of the Streamlit demo with a “not clinical advice” caveat