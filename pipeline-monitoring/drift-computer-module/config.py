import os


def _int(name, default):
    return int(os.environ.get(name, default))


def _float(name, default):
    return float(os.environ.get(name, default))


# Model state is read from the Feast feature server over HTTP only.
FEAST_URL = os.environ.get("FEAST_URL", "http://10.148.0.6:6566")

# Kafka
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "10.148.0.4:9092")
INPUT_TOPIC = os.environ.get("INPUT_TOPIC", "embeddings")
OUTPUT_TOPIC = os.environ.get("OUTPUT_TOPIC", "drift-signals")
GROUP_ID = os.environ.get("GROUP_ID", "drift-computer")

# What we monitor
MODEL_VERSION = os.environ.get("MODEL_VERSION", "bert_768")

# Count-based tumbling window. Must match the window_size the threshold was calibrated with.
WINDOW_SIZE = _int("WINDOW_SIZE", 500)

# Readiness re-check backoff while paused (seconds), with a cap.
BACKOFF_START = _float("BACKOFF_START", 2.0)
BACKOFF_MAX = _float("BACKOFF_MAX", 30.0)

# HTTP timeouts for the Feast readiness call (seconds).
HTTP_TIMEOUT = _float("HTTP_TIMEOUT", 5.0)

# Prometheus /metrics port (the `drift-detector` scrape target).
METRICS_PORT = _int("METRICS_PORT", 8000)

# Kafka poll timeout (seconds).
POLL_TIMEOUT = _float("POLL_TIMEOUT", 1.0)
