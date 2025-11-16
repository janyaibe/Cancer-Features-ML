import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import mlflow
import joblib

data_path = Path("data/processed/features.parquet")
df = pd.read_parquet(data_path)

# Features and label (toy: predict event)
y = df["event"].astype(int)
X = df[["age_at_diagnosis","ajcc_stage","is_erpos","is_prpos","is_her2pos"]].copy()

cat_cols = ["ajcc_stage"]
num_cols = ["age_at_diagnosis","is_erpos","is_prpos","is_her2pos"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols),
])

clf = Pipeline([
    ("prep", preprocess),
    ("lr", LogisticRegression(max_iter=200))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("cancer_features_baseline")

with mlflow.start_run() as run:
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)
    mlflow.log_metric("auc", float(auc))
    mlflow.log_param("model", "logreg_baseline")
    mlflow.log_artifact(str(data_path))

    # Save model artifact
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "baseline_model.pkl"
    joblib.dump(clf, model_path)
    mlflow.log_artifact(str(model_path))

print("Training complete. Model saved to models/baseline_model.pkl")