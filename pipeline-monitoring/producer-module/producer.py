import json
import os
import sys
import time

from google.cloud import bigquery
from confluent_kafka import Producer

PROJECT = os.environ.get("GCP_PROJECT", "ta-driftmon-gana")
DATASET = os.environ.get("BQ_DATASET", "driftmon")
TABLE = os.environ.get("EMBEDDINGS_TABLE", "embeddings")
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "10.148.0.4:9092")
TOPIC = os.environ.get("TOPIC", "embeddings")
# test | train | all
SOURCE_SPLIT = os.environ.get("SOURCE_SPLIT", "test")
# e.g. "0,1,2" or "3"; "" = all
LABEL_FILTER = os.environ.get("LABEL_FILTER", "")
# 0 = no limit
LIMIT = int(os.environ.get("LIMIT", "0"))
# msgs/sec; 0 = unbounded
RATE = float(os.environ.get("RATE", "0"))


def build_query() -> tuple[str, list]:
    where = []
    params = []
    if SOURCE_SPLIT != "all":
        where.append("source_split = @split")
        params.append(bigquery.ScalarQueryParameter("split", "STRING", SOURCE_SPLIT))
    if LABEL_FILTER.strip():
        labels = [int(x) for x in LABEL_FILTER.split(",")]
        where.append("y_original IN UNNEST(@labels)")
        params.append(bigquery.ArrayQueryParameter("labels", "INT64", labels))
    sql = f"SELECT embedding_id, embedding, y_predicted, y_original FROM `{PROJECT}.{DATASET}.{TABLE}`"
    if where:
        sql += " WHERE " + " AND ".join(where)
    # replay in temporal order
    sql += " ORDER BY event_timestamp"
    if LIMIT > 0:
        sql += f" LIMIT {LIMIT}"
    return sql, params


def main() -> int:
    sql, params = build_query()
    print(f"[producer] query: {sql}", flush=True)
    bq = bigquery.Client(project=PROJECT)
    rows = bq.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params)).result()

    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP})
    interval = (1.0 / RATE) if RATE > 0 else 0.0
    n = 0
    for row in rows:
        producer.produce(TOPIC, key=str(row["embedding_id"]), value=json.dumps({
            "embedding_id": row["embedding_id"],
            "embedding": list(row["embedding"]),
            "y_predicted": row["y_predicted"],
            "y_original": row["y_original"],
        }))
        producer.poll(0)
        n += 1
        if n % 1000 == 0:
            print(f"[producer] sent {n}", flush=True)
        if interval:
            time.sleep(interval)
    producer.flush(30)
    print(f"[producer] done — {n} messages to '{TOPIC}'", flush=True)
    return 0



if __name__ == "__main__":
    sys.exit(main())
