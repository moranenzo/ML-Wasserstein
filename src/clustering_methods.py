import numpy as np
import ot
from tqdm.notebook import tqdm

from .utils import computeDistanceMatrix, compute_barycenter_for_cluster

from joblib import Parallel, delayed


# B-WKM : K-Means clustering barycenter-based from Zhuang et al. (2022)

def bary_WKMeans_1d(data, support, n_clusters, n_iter=20, weights=None, random_state=42):
    """
    Wasserstein K-Means barycenter-based clustering by Yubo Zhuang et al. (2022)
    for 1D distributions with same support.

    Parameters:
        data: np.ndarray (n_samples, n_bins)
            Array of histograms (each row is a probability distribution).
        support: np.ndarray (n_bins,)
            The fixed support over which distributions are defined.
        n_clusters: int
            Number of clusters.
        n_iter: int
            Number of iterations.
        weights: np.ndarray
            Weights for the bins. Defaults to uniform.
        random_state: int
            Seed for reproducibility.

    Returns:
        assignments: np.ndarray (n_samples,)
            Cluster assignment for each distribution.
        barycenters: list of np.ndarray
            List of cluster barycenters.
    """
    rng = np.random.default_rng(random_state)
    n_samples = data.shape[0]

    # Random initial cluster assignments
    assignments = rng.integers(low=0, high=n_clusters, size=n_samples)

    for _ in range(n_iter):
        barycenters = []
        for k in range(n_clusters):
            members = data[assignments == k]
            if len(members) == 0:
                barycenters.append(data[rng.integers(n_samples)])
            else:
                barycenters.append(members.mean(axis=0))

        # Reassign
        for i in range(n_samples):
            dists = [np.mean(np.abs(np.sort(data[i]) - np.sort(bary))) for bary in barycenters]
            assignments[i] = int(np.argmin(dists))

    return assignments, barycenters


def bary_WKMeans(data, grid, n_clusters=3, n_iter=20, reg=1e-1, random_state=None):
    """
    Wasserstein K-Means clustering for histograms with shared support.

    Parameters:
        data: np.ndarray of shape (n_samples, n_bins)
            Histograms (distributions projected on the common grid).
        grid: np.ndarray of shape (n_bins, 2)
            Shared support (coordinates of the grid points).
        n_clusters: int
            Number of clusters.
        n_iter: int
            Number of iterations.
        reg: float
            Entropic regularization (for computing barycenters).
        random_state: int or None
            Random seed for reproducibility.

    Returns:
        assignments: np.ndarray of shape (n_samples,)
            Cluster assignments.
        barycenters: list of np.ndarray of shape (n_bins,)
            Cluster barycenters.
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_bins = data.shape

    # Normalize histograms
    data = data / data.sum(axis=1, keepdims=True)

    # Initial barycenters
    indices = rng.choice(n_samples, size=n_clusters, replace=False)
    barycenters = data[indices].copy()

    cost_matrix = ot.dist(grid, grid)
    assignments = np.full(n_samples, -1)  # dummy init for first comparison

    for it in tqdm(range(n_iter), desc="Wasserstein K-Means iterations"):
        # Assignment step
        results = Parallel(n_jobs=-1)(
            delayed(ot.emd2)(data[i], barycenters[k], cost_matrix)
            for i in range(n_samples) for k in range(n_clusters)
        )
        distances = np.array(results).reshape(n_samples, n_clusters)
        new_assignments = distances.argmin(axis=1)

        # Stopping criterion
        if np.array_equal(assignments, new_assignments):
            print(f"Converged at iteration {it}")
            break

        assignments = new_assignments

        # Generate new seeds for barycenters
        seeds = rng.integers(0, 1e9, size=n_clusters)

        # Barycenter update step (parallelized with independent seeds)
        barycenters = Parallel(n_jobs=-1)(
            delayed(compute_barycenter_for_cluster)(
                k, assignments, data, cost_matrix, reg, seeds[k]
            )
            for k in range(n_clusters)
        )

    return assignments, barycenters


# D-WKM : K-Means clustering pairwise-based from Zhuang et al. (2022)

def dist_WKMeans(data, grid, dist_matrix=None, n_clusters=2, n_iter=10, random_state=None):
    rng = np.random.default_rng(random_state)
    n_samples, n_bins = data.shape

    # Normalize the input distributions
    data = data / data.sum(axis=1, keepdims=True)

    if dist_matrix is None:
        dist_matrix = computeDistanceMatrix(data, grid)

    # Initialize assignments randomly
    assignments = rng.integers(0, n_clusters, size=n_samples)

    for it in tqdm(range(n_iter), desc="Wasserstein pairwise clustering iterations"):
        new_assignments = np.zeros_like(assignments)

        for i in range(n_samples):
            avg_dists = np.array([
                dist_matrix[i, assignments == k].mean() if np.any(assignments == k) else np.inf
                for k in range(n_clusters)
            ])
            new_assignments[i] = np.argmin(avg_dists)

        for k in range(n_clusters):
            if not np.any(new_assignments == k):
                new_assignments[rng.integers(0, n_samples)] = k

        if np.array_equal(assignments, new_assignments):
            print(f"Converged at iteration {it}")
            break

        assignments = new_assignments

    return assignments


# W-SDP : Semidefinite Program relaxation of the distance-based K-means
