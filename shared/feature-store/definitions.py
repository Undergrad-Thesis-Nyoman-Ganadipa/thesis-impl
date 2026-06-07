from datetime import timedelta

from feast import Entity, FeatureView, Field, BigQuerySource
from feast.types import Array, Float64, Int64, String

# Embeddings: input features, read offline to build the fitting dataset.
embedding = Entity(name="embedding", join_keys=["embedding_id"])

embeddings_source = BigQuerySource(
    name="embeddings_source",
    table="ta-driftmon-gana.driftmon.embeddings",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

embeddings_fv = FeatureView(
    name="embeddings",
    entities=[embedding],
    ttl=timedelta(days=3650),
    online=False,
    source=embeddings_source,
    schema=[
        Field(name="embedding",    dtype=Array(Float64)),
        Field(name="y_predicted",  dtype=Int64),
        Field(name="y_original",   dtype=Int64),
        Field(name="source_split", dtype=String),
    ],
)

# Baseline: model state, online-served, keyed by (model_version, baseline_id).
model = Entity(name="model", join_keys=["model_version"])
baseline_version = Entity(name="baseline_version", join_keys=["baseline_id"])

baselines_source = BigQuerySource(
    name="baselines_source",
    table="ta-driftmon-gana.driftmon.baselines",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

drift_baseline_fv = FeatureView(
    name="drift_baseline",
    # composite: (model_version, baseline_id)
    entities=[model, baseline_version],
    # baselines are long-lived
    ttl=timedelta(days=3650),
    online=True,
    source=baselines_source,
    schema=[
        Field(name="reference_set_id",     dtype=String),
        Field(name="batch_n_pc",           dtype=Int64),
        Field(name="n_samples",            dtype=Int64),
        Field(name="threshold",            dtype=Float64),
        # (batch_n_pc,)
        Field(name="mean",                 dtype=Array(Float64)),
        # (batch_n_pc^2,)
        Field(name="cov",                  dtype=Array(Float64)),
        Field(name="cov_shape",            dtype=Array(Int64)),
        # (batch_n_pc*dim,)
        Field(name="pca_components",       dtype=Array(Float64)),
        Field(name="pca_components_shape", dtype=Array(Int64)),
        # (dim,)
        Field(name="pca_mean",             dtype=Array(Float64)),
    ],
)

# Active pointer: which baseline_id is current, latest-wins by event_timestamp.
active_pointer_source = BigQuerySource(
    name="active_pointer_source",
    table="ta-driftmon-gana.driftmon.active_pointer",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

active_baseline_fv = FeatureView(
    name="active_baseline",
    entities=[model],
    ttl=timedelta(days=3650),
    online=True,
    source=active_pointer_source,
    schema=[
        Field(name="active_baseline_id", dtype=String),
    ],
)
