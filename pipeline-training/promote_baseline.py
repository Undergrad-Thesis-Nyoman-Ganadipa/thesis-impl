import json
import datetime

from google.cloud import bigquery

import config

TABLE = f"{config.GCP_PROJECT_ID}.{config.BQ_DATASET}.active_pointer"

SCHEMA = [
    bigquery.SchemaField("model_version", "STRING"),
    bigquery.SchemaField("active_baseline_id", "STRING"),
    bigquery.SchemaField("event_timestamp", "TIMESTAMP"),
    bigquery.SchemaField("created_timestamp", "TIMESTAMP"),
]


def main(baseline_id=None):
    if baseline_id is None:
        meta = json.loads((config.ARTIFACTS_DIR / "baseline_meta.json").read_text())
        baseline_id = meta["baseline_id"]

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    row = {
        "model_version":      config.MODEL_VERSION,
        "active_baseline_id": baseline_id,
        "event_timestamp":    now,
        "created_timestamp":  now,
    }

    client = bigquery.Client(project=config.GCP_PROJECT_ID, location=config.BQ_LOCATION)
    client.load_table_from_json(
        [row], TABLE,
        job_config=bigquery.LoadJobConfig(schema=SCHEMA, write_disposition="WRITE_APPEND"),
    ).result()

    print(f"Promoted active baseline: model_version={config.MODEL_VERSION} "
          f"-> active_baseline_id={baseline_id}")


if __name__ == "__main__":
    main()
