import hashlib
import json

import config


def fit_config() -> dict:
    """Everything (besides the reference data) that determines the artifact."""
    return {
        "model_version":         config.MODEL_VERSION,
        "reference_set_id":      config.REFERENCE_SET_ID,
        "label_list":            list(config.TRAINING_LABELS),
        "batch_n_pc":            config.BATCH_N_PC,
        "per_label_n_pc":        config.PER_LABEL_N_PC,
        "window_size":           config.WINDOW_SIZE,
        "n_th_samples":          config.N_TH_SAMPLES,
        "threshold_sensitivity": config.THRESHOLD_SENSITIVITY,
        "seed":                  config.SEED,
    }


def compute_baseline_id(train_hash: str, test_hash: str) -> str:
    """16-hex-char (64-bit) content id over reference hashes + fit config."""
    payload = {"train_hash": train_hash, "test_hash": test_hash, **fit_config()}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
