import json
import datetime

import numpy as np
from google.cloud import bigquery

import config

TABLE = f"{config.GCP_PROJECT_ID}.{config.BQ_DATASET}.baselines"

SCHEMA = [
    bigquery.SchemaField("model_version", "STRING"),
    bigquery.SchemaField("baseline_id", "STRING"),
    bigquery.SchemaField("reference_set_id", "STRING"),
    bigquery.SchemaField("event_timestamp", "TIMESTAMP"),
    bigquery.SchemaField("created_timestamp", "TIMESTAMP"),
    bigquery.SchemaField("batch_n_pc", "INT64"),
    bigquery.SchemaField("n_samples", "INT64"),
    bigquery.SchemaField("threshold", "FLOAT64"),
    bigquery.SchemaField("mean", "FLOAT64", mode="REPEATED"),
    bigquery.SchemaField("cov", "FLOAT64", mode="REPEATED"),
    bigquery.SchemaField("cov_shape", "INT64", mode="REPEATED"),
    bigquery.SchemaField("pca_components", "FLOAT64", mode="REPEATED"),
    bigquery.SchemaField("pca_components_shape", "INT64", mode="REPEATED"),
    bigquery.SchemaField("pca_mean", "FLOAT64", mode="REPEATED"),
]


def main():
    npz  = np.load(config.ARTIFACTS_DIR / "baseline_batch.npz")
    meta = json.loads((config.ARTIFACTS_DIR / "baseline_meta.json").read_text())

    mean = npz["mean"]
    cov  = npz["cov"]
    pca_components = npz["pca_components"]
    pca_mean = npz["pca_mean"]

    now = datetime.datetime.now(datetime.timezone.utc)
    now_iso = now.isoformat()
    row = {
        "model_version":        meta["model_version"],
        "baseline_id":          meta["baseline_id"],
        "reference_set_id":     meta["reference_set_id"],
        "event_timestamp":      now_iso,
        "created_timestamp":    now_iso,
        "batch_n_pc":           int(meta["batch_n_pc"]),
        "n_samples":            int(meta["n_samples"]),
        "threshold":            float(meta["threshold"]),
        "mean":                 mean.ravel().astype(float).tolist(),
        "cov":                  cov.ravel(order="C").astype(float).tolist(),
        "cov_shape":            list(cov.shape),
        "pca_components":       pca_components.ravel(order="C").astype(float).tolist(),
        "pca_components_shape": list(pca_components.shape),
        "pca_mean":             pca_mean.ravel().astype(float).tolist(),
    }

    client = bigquery.Client(project=config.GCP_PROJECT_ID, location=config.BQ_LOCATION)
    job = client.load_table_from_json(
        [row], TABLE,
        job_config=bigquery.LoadJobConfig(schema=SCHEMA, write_disposition="WRITE_APPEND"),
    )
    job.result()

    n = client.get_table(TABLE).num_rows
    print(f"Published baseline for model_version={row['model_version']} "
          f"baseline_id={row['baseline_id']} reference_set_id={row['reference_set_id']}")
    print(f"  mean {mean.shape}  cov {cov.shape}  pca_components {pca_components.shape}")
    print(f"  threshold={row['threshold']:.4f}  event_timestamp={now.isoformat()}")
    print(f"  {TABLE} now has {n} row(s)")


if __name__ == "__main__":
    main()
