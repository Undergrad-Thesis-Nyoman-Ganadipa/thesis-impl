from pathlib import Path

import numpy as np
from feast import FeatureStore

import baseline_serving

_HERE = Path(__file__).resolve().parent
ARTIFACTS = _HERE.parent.parent / "pipeline-training" / "artifacts" / "baseline_batch.npz"

FEATURES = [
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


def main():
    model_version = "bert_768"
    store = FeatureStore(repo_path=str(_HERE))

    # Two-step: resolve the active pointer, then read THAT exact version.
    active_id = baseline_serving.resolve_active_baseline_id(store, model_version)
    if active_id is None:
        print("NOT SERVED — no active baseline pointer for "
              f"'{model_version}'. Run the training pipeline / materialize first.")
        return 1
    print(f"active baseline_id = {active_id}")

    r = store.get_online_features(
        features=FEATURES,
        entity_rows=[{"model_version": model_version, "baseline_id": active_id}],
    ).to_dict()

    mean  = np.asarray(r["mean"][0])
    cov   = np.asarray(r["cov"][0]).reshape(r["cov_shape"][0])
    pca_c = np.asarray(r["pca_components"][0]).reshape(r["pca_components_shape"][0])
    pca_m = np.asarray(r["pca_mean"][0])

    print("=== read from Feast online store ===")
    print(f"  reference_set_id : {r['reference_set_id'][0]}")
    print(f"  threshold        : {r['threshold'][0]:.4f}")
    print(f"  batch_n_pc       : {r['batch_n_pc'][0]}   n_samples: {r['n_samples'][0]}")
    print(f"  mean {mean.shape}  cov {cov.shape}  pca_components {pca_c.shape}  pca_mean {pca_m.shape}")

    npz = np.load(ARTIFACTS)
    checks = {
        "mean":           np.allclose(mean,  npz["mean"],  atol=1e-9),
        "cov":            np.allclose(cov,   npz["cov"],   atol=1e-9),
        "pca_components": np.allclose(pca_c, npz["pca_components"], atol=1e-9),
        "pca_mean":       np.allclose(pca_m, npz["pca_mean"], atol=1e-9),
    }
    print("\n=== round-trip vs fitted artifact ===")
    for k, ok in checks.items():
        print(f"  {k:16} {'ok' if ok else 'MISMATCH'}")

    ok = all(checks.values())
    print("\n" + ("PASS — Feast serves the baseline losslessly" if ok else "FAIL — mismatch"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
