import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
# .../TA
_TA = _HERE.parents[2]
_DRIFTLENS = _TA / "drift-lens"
_ARTIFACTS = _HERE.parents[1] / "pipeline-training" / "artifacts"

sys.path.insert(0, str(_DRIFTLENS))
from driftlens.driftlens import DriftLens                       # noqa: E402
from driftlens.distribution_distances import frechet_drift_distance as ref_fdd  # noqa: E402

from fdd import compute_fdd                                     # noqa: E402


def main() -> int:
    npz = np.load(_ARTIFACTS / "baseline_batch.npz")
    baseline_dict = {
        "mean": npz["mean"], "cov": npz["cov"],
        "pca_components": npz["pca_components"], "pca_mean": npz["pca_mean"],
    }

    dl = DriftLens()
    bl = dl.load_baseline(str(_ARTIFACTS), "baseline")

    # 1) stored stats == native baseline object stats
    pca = bl.get_batch_PCA_model()
    checks = {
        "mean":           np.allclose(baseline_dict["mean"], bl.get_batch_mean_vector()),
        "cov":            np.allclose(baseline_dict["cov"], bl.get_batch_covariance_matrix()),
        "pca_components": np.allclose(baseline_dict["pca_components"], pca.components_),
        "pca_mean":       np.allclose(baseline_dict["pca_mean"], pca.mean_),
    }

    # 2) per-batch FDD parity on a fixed random window
    dim = baseline_dict["pca_mean"].shape[0]
    E_w = np.random.RandomState(0).randn(500, dim)

    ours = compute_fdd(E_w, baseline_dict)

    # DriftLens uses sklearn PCA.transform
    E_red = pca.transform(E_w)
    ref = float(ref_fdd.frechet_distance(
        bl.get_batch_mean_vector(), ref_fdd.get_mean(E_red),
        bl.get_batch_covariance_matrix(), ref_fdd.get_covariance(E_red),
    ))

    print("=== stored stats vs native baseline ===")
    for k, ok in checks.items():
        print(f"  {k:16} {'ok' if ok else 'MISMATCH'}")
    print("=== per-batch FDD ===")
    print(f"  ours = {ours:.10f}")
    print(f"  ref  = {ref:.10f}")
    fdd_ok = np.isclose(ours, ref, rtol=1e-9, atol=1e-9)
    print(f"  parity: {'ok' if fdd_ok else 'MISMATCH'}")

    ok = all(checks.values()) and fdd_ok
    print("\n" + ("PASS — detector FDD is identical to DriftLens" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
