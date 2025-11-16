from feast import Entity

patient = Entity(
    name="patient_id",
    join_keys=["patient_id"],
    description="Patient identifier",
)