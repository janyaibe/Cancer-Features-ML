from datetime import timedelta
from feast import FeatureView, Field, FileSource
from feast.types import Int64, String, Bool, Float32
from feast import Entity

# Define the data source (features parquet)
features_source = FileSource(
    path="data/processed/features.parquet",
    timestamp_field="event_timestamp",
)

patient = Entity(name="patient_id", join_keys=["patient_id"], description="Patient identifier")

patient_diagnosis_features = FeatureView(
    name="patient_diagnosis_features",
    entities=[patient],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="age_at_diagnosis", dtype=Float32),
        Field(name="age_bin", dtype=String),
        Field(name="ajcc_stage", dtype=String),
        Field(name="is_erpos", dtype=Int64),
        Field(name="is_prpos", dtype=Int64),
        Field(name="is_her2pos", dtype=Int64),
    ],
    source=features_source,
    online=True,
)