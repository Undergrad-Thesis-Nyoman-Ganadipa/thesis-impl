import numpy as np
import h5py

import config
import bq_reader

HDF5_FILES = {
    "train": "train_embedding_0_1_2_layer_12.hdf5",
    "test":  "test_embedding_0_1_2_layer_12.hdf5",
}


def load_hdf5(split):
    with h5py.File(config.HDF5_DIR / HDF5_FILES[split], "r") as hf:
        E     = hf["E"][()].astype(np.float64)
        Ypred = hf["Y_predicted"][()].astype(np.int64)
    return E, Ypred


def main():
    print("=" * 60)
    print("  Verify — BigQuery vs source HDF5")
    print("=" * 60)

    all_ok = True
    for split in ["train", "test"]:
        E_bq, Y_bq = bq_reader.read_reference(split)
        E_h,  Y_h  = load_hdf5(split)

        shape_ok = E_bq.shape == E_h.shape
        emb_ok   = shape_ok and np.allclose(E_bq, E_h, atol=1e-6, rtol=0)
        y_ok     = shape_ok and np.array_equal(Y_bq, Y_h)
        max_diff = float(np.abs(E_bq - E_h).max()) if shape_ok else float("nan")

        print(f"\n  [{split}]")
        print(f"    shape       BQ {E_bq.shape}  HDF5 {E_h.shape}  -> {'ok' if shape_ok else 'MISMATCH'}")
        print(f"    embeddings  max|diff|={max_diff:.2e}  -> {'ok' if emb_ok else 'MISMATCH'}")
        print(f"    pred labels -> {'ok' if y_ok else 'MISMATCH'}")

        all_ok = all_ok and emb_ok and y_ok

    print("\n" + "-" * 60)
    print("  PASS — round-trip lossless" if all_ok else "  FAIL — data mismatch")
    print("-" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
