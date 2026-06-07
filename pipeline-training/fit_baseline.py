import sys
import json

import numpy as np

import config
sys.path.insert(0, str(config.DRIFTLENS_ROOT))
from driftlens.driftlens import DriftLens

import bq_reader
import identity


def compute_threshold(sorted_distances, sensitivity):
    l = np.asarray(sorted_distances)
    lo = np.quantile(l, sensitivity / 100)
    hi = np.quantile(l, (100 - sensitivity) / 100)
    l = l[(l > lo) & (l < hi)]
    return float(l.max())


def main(baseline_id=None, hashes=None):
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # DriftLens seeds the global numpy RNG; seed once for a reproducible fit
    np.random.seed(config.SEED)

    dl = DriftLens()

    print("Reading train from BigQuery...")
    E_train, Y_train = bq_reader.read_reference("train")
    print(f"  train E={E_train.shape}")

    print("Estimating baseline (offline)...")
    baseline = dl.estimate_baseline(
        E=E_train, Y=Y_train,
        label_list=config.TRAINING_LABELS,
        batch_n_pc=config.BATCH_N_PC,
        per_label_n_pc=config.PER_LABEL_N_PC,
    )

    print("Reading test from BigQuery...")
    E_test, Y_test = bq_reader.read_reference("test")
    print(f"  test E={E_test.shape}")

    print("Estimating threshold...")


    with np.errstate(all="ignore"):
        per_batch_sorted, _ = dl.random_sampling_threshold_estimation(
            label_list=config.TRAINING_LABELS,
            E=E_test, Y=Y_test,
            batch_n_pc=config.BATCH_N_PC,
            per_label_n_pc=config.PER_LABEL_N_PC,
            window_size=config.WINDOW_SIZE,
            n_samples=config.N_TH_SAMPLES,
            flag_shuffle=True,
            flag_replacement=True,
        )
    threshold = compute_threshold(per_batch_sorted, config.THRESHOLD_SENSITIVITY)
    print(f"  threshold={threshold:.4f}")

    # DriftLens-native artifact (full baseline)
    dl.save_baseline(str(config.ARTIFACTS_DIR), "baseline")

    # per-batch state that Feast will hold
    pca = baseline.PCA_models_dict["batch"]
    np.savez(
        config.ARTIFACTS_DIR / "baseline_batch.npz",
        mean=baseline.mean_vectors_dict["batch"],
        cov=baseline.covariance_matrices_dict["batch"],
        pca_components=pca.components_,
        pca_mean=pca.mean_,
    )
    # baseline_id = hash(reference data + fit config + seed)
    if hashes is None:
        hashes = bq_reader.reference_hashes()
    if baseline_id is None:
        baseline_id = identity.compute_baseline_id(hashes["train"], hashes["test"])
    print(f"  baseline_id={baseline_id}  (train={hashes['train'][:12]}… test={hashes['test'][:12]}…)")

    meta = {
        "model_version":    config.MODEL_VERSION,
        "reference_set_id": config.REFERENCE_SET_ID,
        "baseline_id":      baseline_id,
        "train_hash":       hashes["train"],
        "test_hash":        hashes["test"],
        "batch_n_pc":       config.BATCH_N_PC,
        "n_samples":        int(baseline.n_samples_dict["batch"]),
        "threshold":        threshold,
        "window_size":      config.WINDOW_SIZE,
        "label_list":       config.TRAINING_LABELS,
        "seed":             config.SEED,
    }
    with open(config.ARTIFACTS_DIR / "baseline_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved -> {config.ARTIFACTS_DIR}")
    print("  baseline/            (DriftLens native)")
    print("  baseline_batch.npz   (mean, cov, pca_components, pca_mean)")
    print("  baseline_meta.json   (threshold + metadata)")


if __name__ == "__main__":
    main()
