.PHONY: setup data features validate train app feast-registry

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

data:
	python src/ingest/prepare_processed.py

features: data feast-registry
	python src/features/build_features.py

feast-registry:
	feast apply -c feature_repo

validate:
	python src/validation/run_checks.py

train: features
	python src/training/train_baseline.py

app:
	streamlit run app/streamlit_app.py