from pathlib import Path

import yaml

_PIPELINE_DIR    = Path(__file__).resolve().parent
THESIS_IMPL_ROOT = _PIPELINE_DIR.parent
TA_ROOT          = THESIS_IMPL_ROOT.parent

with open(THESIS_IMPL_ROOT / "config.yaml") as f:
    _cfg = yaml.safe_load(f)

# Google cloud
GCP_PROJECT_ID = _cfg["gcp"]["project_id"]
BQ_DATASET     = _cfg["gcp"]["dataset"]
BQ_LOCATION    = _cfg["gcp"]["location"]

# Model / reference set
MODEL_VERSION    = _cfg["model"]["version"]
EMBEDDING_DIM    = _cfg["model"]["embedding_dim"]
LAYER            = _cfg["model"]["layer"]
REFERENCE_SET_ID = _cfg["model"]["reference_set_id"]

# DriftLens params
_dl = _cfg["driftlens"]
TRAINING_LABELS       = _dl["training_labels"]
DRIFT_LABELS          = _dl["drift_labels"]
BATCH_N_PC            = _dl["batch_n_pc"]
PER_LABEL_N_PC        = _dl["per_label_n_pc"]
WINDOW_SIZE           = _dl["window_size"]
N_TH_SAMPLES          = _dl["n_th_samples"]
THRESHOLD_SENSITIVITY = _dl["threshold_sensitivity"]
SEED                  = _dl["seed"]

# Paths
DRIFTLENS_ROOT = TA_ROOT / "drift-lens"
HDF5_DIR       = TA_ROOT / "small-prototype/experiments/driftlens/use_case_1_1/data/embeddings/2/layer-12"
ARTIFACTS_DIR  = _PIPELINE_DIR / "artifacts"
FEATURE_STORE_REPO = THESIS_IMPL_ROOT / "shared" / "feature-store"

FQ_EMBEDDINGS = f"`{GCP_PROJECT_ID}.{BQ_DATASET}.embeddings`"
FQ_MEMBERS    = f"`{GCP_PROJECT_ID}.{BQ_DATASET}.reference_set_members`"
