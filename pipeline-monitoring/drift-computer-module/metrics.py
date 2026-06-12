import threading
from collections import defaultdict

from prometheus_client import Counter, Gauge, start_http_server
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import REGISTRY

drift_windows_processed_total = Counter(
    "drift_windows_processed_total", "Windows scored (FDD computed)",
)
drift_detected_total = Counter(
    "drift_detected_total", "Windows whose FDD exceeded the threshold",
)

# Served baseline's drift threshold; a plain held gauge.
drift_threshold = Gauge(
    "drift_threshold", "Drift threshold — FDD above this is flagged as drift",
    ["model_version", "reference_set_id"],
)

# Readiness as a one-hot gauge: current reason is 1, others 0.
_READINESS_REASONS = ("served", "baseline_missing", "redis_down", "feast_server_down")
detector_readiness = Gauge(
    "detector_readiness", "1 = current readiness state, 0 = not", ["reason"],
)


def set_readiness(reason: str) -> None:
    """Set the given reason to 1 and all other known reasons to 0."""
    for r in _READINESS_REASONS:
        detector_readiness.labels(reason=r).set(1.0 if r == reason else 0.0)


class _DriftScoreAggregator:
    """Collector for drift_fdd_score: mean of windows scored since the last
    scrape, per (model_version, reference_set_id); absent when none -> "no data".

    `collect()` is invoked by the exposition layer on every scrape; it drains the
    buffer, so each scraped sample covers exactly one scrape interval. Query the
    value through Prometheus, not by curling /metrics directly (a manual scrape
    would drain the buffer and rob the next Prometheus scrape)."""

    def __init__(self):
        self._lock = threading.Lock()
        # list of (fdd, model_version, reference_set_id)
        self._scores = []

    def record(self, fdd: float, model_version: str, reference_set_id: str) -> None:
        with self._lock:
            self._scores.append((float(fdd), model_version, reference_set_id))

    def collect(self):
        with self._lock:
            scores, self._scores = self._scores, []
        if not scores:
            # emit nothing -> series absent -> Grafana shows no-data
            return
        groups = defaultdict(list)
        for fdd, mv, ref in scores:
            groups[(mv, ref)].append(fdd)
        fam = GaugeMetricFamily(
            "drift_fdd_score",
            "Fréchet Drift Distance of windows in the scrape interval (agg=avg|max)",
            labels=["model_version", "reference_set_id", "agg"],
        )
        for (mv, ref), vals in groups.items():
            fam.add_metric([mv, ref, "avg"], sum(vals) / len(vals))
            fam.add_metric([mv, ref, "max"], max(vals))
        yield fam


drift_score = _DriftScoreAggregator()


def serve(port: int) -> None:
    # register the custom collector before serving
    REGISTRY.register(drift_score)
    start_http_server(port)
