import numpy as np
import scipy.linalg

# vendored verbatim from DriftLens; _EPS regularises the diagonal
_EPS = 1e-6


def get_covariance(E) -> np.ndarray:
    return np.cov(E, rowvar=False)


def get_mean(E) -> np.ndarray:
    return E.mean(0)


def matrix_sqrt(X) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.real(scipy.linalg.sqrtm(X))


def frechet_distance(mu_x, mu_y, sigma_x, sigma_y) -> float:
    n = sigma_x.shape[0]
    reg = _EPS * np.eye(n)
    with np.errstate(all="ignore"):
        product = (sigma_x + reg) @ (sigma_y + reg)
        sqrt_term = matrix_sqrt(product)
    return np.linalg.norm(mu_x - mu_y) + np.trace(sigma_x + sigma_y - 2 * sqrt_term)


def pca_transform(E_w, pca_components, pca_mean) -> np.ndarray:
    return (E_w - pca_mean) @ pca_components.T


def compute_fdd(E_w, baseline) -> float:
    E_reduced = pca_transform(E_w, baseline["pca_components"], baseline["pca_mean"])
    mean_w = get_mean(E_reduced)
    cov_w = get_covariance(E_reduced)
    return float(frechet_distance(baseline["mean"], mean_w, baseline["cov"], cov_w))
