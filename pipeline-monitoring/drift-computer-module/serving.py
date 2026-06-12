import requests

# Feature fields read for the active baseline (the exact version, composite key).
_BASELINE_FIELDS = [
    "drift_baseline:reference_set_id",
    "drift_baseline:threshold",
    "drift_baseline:batch_n_pc",
    "drift_baseline:n_samples",
    "drift_baseline:mean",
    "drift_baseline:cov",
    "drift_baseline:cov_shape",
    "drift_baseline:pca_components",
    "drift_baseline:pca_components_shape",
    "drift_baseline:pca_mean",
]


class _Down(Exception):
    """Raised for 5xx (redis_down) or connection failure (feast_server_down)."""
    def __init__(self, reason):
        self.reason = reason


def _get_online(feast_url, features, entities, timeout):
    """POST /get-online-features -> dict {feature_name: value}. Raises _Down on
    5xx (redis behind server) or connection error/timeout (server down)."""
    try:
        resp = requests.post(
            f"{feast_url}/get-online-features",
            json={"features": features, "entities": entities},
            timeout=timeout,
        )
    except (requests.ConnectionError, requests.Timeout):
        raise _Down("feast_server_down")
    if resp.status_code >= 500:
        raise _Down("redis_down")
    resp.raise_for_status()
    body = resp.json()
    names = body["metadata"]["feature_names"]
    # results[i].values[0] aligns with names[i]; single entity row -> index 0.
    return {n: body["results"][i]["values"][0] for i, n in enumerate(names)}


def resolve_active_baseline_id(feast_url, model_version, timeout):
    d = _get_online(
        feast_url, ["active_baseline:active_baseline_id"],
        {"model_version": [model_version]}, timeout,
    )
    # None if pointer absent (200 + null)
    return d.get("active_baseline_id")


def read_baseline(feast_url, model_version, baseline_id, timeout):
    return _get_online(
        feast_url, _BASELINE_FIELDS,
        {"model_version": [model_version], "baseline_id": [baseline_id]}, timeout,
    )


def baseline_readiness(feast_url, model_version, timeout):
    """Return (reason, raw_or_None).

    reason in {served, baseline_missing, redis_down, feast_server_down}.
    raw is the feature dict (only when 'served'), to be reshaped by the caller.
    Never raises on infra failure — converts it to a reason.
    """
    try:
        active_id = resolve_active_baseline_id(feast_url, model_version, timeout)
        if active_id is None:
            return "baseline_missing", None
        raw = read_baseline(feast_url, model_version, active_id, timeout)
        # status lies; the VALUE is the source of truth for a composite-key miss.
        if raw.get("n_samples") is None:
            return "baseline_missing", None
        raw["_active_baseline_id"] = active_id
        return "served", raw
    except _Down as d:
        return d.reason, None
