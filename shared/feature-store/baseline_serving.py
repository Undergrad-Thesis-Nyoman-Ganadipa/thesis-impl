from pathlib import Path

from feast import FeatureStore

# redis-py is a Feast dependency; guard the import defensively
try:
    from redis.exceptions import (
        ConnectionError as _RedisConnectionError,
        TimeoutError as _RedisTimeoutError,
    )
    _REDIS_ERRORS = (_RedisConnectionError, _RedisTimeoutError)
except Exception:  # pragma: no cover
    _REDIS_ERRORS = ()

_HERE = Path(__file__).resolve().parent


def resolve_active_baseline_id(store: FeatureStore, model_version: str):
    """Read the active pointer -> active_baseline_id (None if no pointer served)."""
    ptr = store.get_online_features(
        features=["active_baseline:active_baseline_id"],
        entity_rows=[{"model_version": model_version}],
    ).to_dict()
    return ptr["active_baseline_id"][0]


def baseline_status(store: FeatureStore, model_version: str):
    """Three-state readiness, never raising on infra failure.

    Returns (status, active_baseline_id) where status is one of:
      'served'      — pointer resolves AND that version's value is non-null
      'missing'     — Redis up but pointer or version absent (re-materialize)
      'unreachable' — Redis down / connection error (restart Redis)
    """
    try:
        active_id = resolve_active_baseline_id(store, model_version)
        if active_id is None:
            return "missing", None
        served = store.get_online_features(
            features=["drift_baseline:n_samples"],
            entity_rows=[{"model_version": model_version, "baseline_id": active_id}],
        ).to_dict()
        if served["n_samples"][0] is None:
            return "missing", active_id
        return "served", active_id
    except _REDIS_ERRORS:
        return "unreachable", None


def is_baseline_served(store: FeatureStore, model_version: str) -> bool:
    """True iff the active baseline is actually present in the online store.
    Null-checks the VALUE (status lies) and treats unreachable as not served."""
    status, _ = baseline_status(store, model_version)
    return status == "served"


def main():
    model_version = "bert_768"
    store = FeatureStore(repo_path=str(_HERE))
    status, active_id = baseline_status(store, model_version)
    if status == "served":
        print(f"READY — baseline '{active_id}' for '{model_version}' is served")
        return 0
    if status == "missing":
        print(f"NOT READY — no served baseline for '{model_version}' "
              f"(active_id={active_id}). Re-materialize before serving.")
        return 1
    print(f"NOT READY — online store UNREACHABLE (Redis down) for '{model_version}'.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
