import datetime

from feast import FeatureStore

import config
import bq_reader
import identity
import fit_baseline
import publish_baseline
import promote_baseline

# Wide lower bound for the ranged materialize (ignores the registry watermark).
_MATERIALIZE_START = datetime.datetime(2000, 1, 1)


def _online(store, features, entity_row):
    return store.get_online_features(features=features, entity_rows=[entity_row]).to_dict()


def is_active_and_served(store, baseline_id):
    """True iff baseline_id is the active pointer and that version is in Redis."""
    ptr = _online(store, ["active_baseline:active_baseline_id"],
                  {"model_version": config.MODEL_VERSION})
    if ptr["active_baseline_id"][0] != baseline_id:
        return False
    served = _online(store, ["drift_baseline:n_samples"],
                     {"model_version": config.MODEL_VERSION, "baseline_id": baseline_id})
    return served["n_samples"][0] is not None


def main():
    store = FeatureStore(repo_path=str(config.FEATURE_STORE_REPO))

    h = bq_reader.reference_hashes()
    baseline_id = identity.compute_baseline_id(h["train"], h["test"])
    print(f"baseline_id={baseline_id}  "
          f"(train={h['train'][:12]}… test={h['test'][:12]}…)")

    if is_active_and_served(store, baseline_id):
        print(f"Baseline {baseline_id} is already the active pointer and served "
              "online — nothing to do.")
        return

    print("\n[1/4] fit baseline")
    fit_baseline.main(baseline_id=baseline_id, hashes=h)

    print("\n[2/4] publish baseline -> BigQuery")
    publish_baseline.main()

    print("\n[3/4] promote active pointer -> BigQuery")
    promote_baseline.main(baseline_id=baseline_id)

    print("\n[4/4] materialize (ranged) -> Redis")
    store.materialize(start_date=_MATERIALIZE_START, end_date=datetime.datetime.now())

    print(f"\nPipeline complete: fit -> publish -> promote -> materialize  "
          f"(baseline_id={baseline_id})")


if __name__ == "__main__":
    main()
