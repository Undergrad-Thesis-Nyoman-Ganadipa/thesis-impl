import hashlib

import numpy as np
import pandas as pd
from google.cloud import bigquery
from feast import FeatureStore

import config

FEATURES = ["embeddings:embedding", "embeddings:y_predicted"]

_bq = None
_store = None


def bq():
    global _bq
    if _bq is None:
        _bq = bigquery.Client(project=config.GCP_PROJECT_ID, location=config.BQ_LOCATION)
    return _bq


def store():
    global _store
    if _store is None:
        _store = FeatureStore(repo_path=str(config.FEATURE_STORE_REPO))
    return _store


def _to_arrays(df):
    idx = df["embedding_id"].str.rsplit("_", n=1).str[-1].astype(int)
    df = df.assign(_i=idx).sort_values("_i", kind="mergesort")
    E = np.asarray(df["embedding"].tolist(), dtype=np.float64)
    Y = df["y_predicted"].to_numpy(dtype=np.int64)
    return E, Y


def _ids(sql, params):
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    rows = bq().query(sql, job_config=job_config).result()
    return [r["embedding_id"] for r in rows]


def _fetch(entity_ids):
    # single as-of timestamp; each embedding_id is unique so the join is 1:1
    entity_df = pd.DataFrame({
        "embedding_id": entity_ids,
        "event_timestamp": pd.Timestamp.now(tz="UTC"),
    })
    df = store().get_historical_features(entity_df=entity_df, features=FEATURES).to_df()
    return _to_arrays(df)


def read_reference(split):
    """Reference data (split='train' or 'test') for the configured reference set."""
    ids = _ids(
        f"""SELECT embedding_id FROM {config.FQ_MEMBERS}
            WHERE reference_set_id = @ref AND split = @split""",
        [bigquery.ScalarQueryParameter("ref", "STRING", config.REFERENCE_SET_ID),
         bigquery.ScalarQueryParameter("split", "STRING", split)],
    )
    return _fetch(ids)


def read_stream(source_split):
    """Stream data (source_split='new_unseen' or 'drift')."""
    ids = _ids(
        f"""SELECT embedding_id FROM {config.FQ_EMBEDDINGS}
            WHERE model_version = @mv AND source_split = @ss""",
        [bigquery.ScalarQueryParameter("mv", "STRING", config.MODEL_VERSION),
         bigquery.ScalarQueryParameter("ss", "STRING", source_split)],
    )
    return _fetch(ids)


def _hash_ids(ids):
    """sha256 over the sorted embedding_ids — same convention as step3b."""
    h = hashlib.sha256()
    for eid in sorted(ids):
        h.update(eid.encode())
    return h.hexdigest()


def reference_hashes():
    """Recompute (train_hash, test_hash) from the CURRENT reference_set_members."""
    out = {}
    for split in ("train", "test"):
        ids = _ids(
            f"""SELECT embedding_id FROM {config.FQ_MEMBERS}
                WHERE reference_set_id = @ref AND split = @split""",
            [bigquery.ScalarQueryParameter("ref", "STRING", config.REFERENCE_SET_ID),
             bigquery.ScalarQueryParameter("split", "STRING", split)],
        )
        out[split] = _hash_ids(ids)
    return out


def reference_set_id_for(train_hash, test_hash):
    """Find which registered reference_set has these hashes (content-addressed).
    Returns the reference_set_id, or None if the current data matches none."""
    sql = f"""
        SELECT reference_set_id
        FROM `{config.GCP_PROJECT_ID}.{config.BQ_DATASET}.reference_sets`
        WHERE train_hash = @th AND test_hash = @sh
        LIMIT 1
    """
    rows = list(bq().query(sql, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("th", "STRING", train_hash),
        bigquery.ScalarQueryParameter("sh", "STRING", test_hash),
    ])).result())
    return rows[0]["reference_set_id"] if rows else None


def baseline_exists_for_ref(reference_set_id):
    """True if a baseline already exists for this reference set + model."""
    sql = f"""
        SELECT COUNT(*) AS n
        FROM `{config.GCP_PROJECT_ID}.{config.BQ_DATASET}.baselines`
        WHERE model_version = @mv AND reference_set_id = @ref
    """
    rows = list(bq().query(sql, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("mv", "STRING", config.MODEL_VERSION),
        bigquery.ScalarQueryParameter("ref", "STRING", reference_set_id),
    ])).result())
    return rows[0]["n"] > 0
