import numpy as np
import ot
from ot.bregman import barycenter as ot_barycenter


# K-Means clustering barycenter-based from Zhuang et al. (2022)

def wbarycenter_clustering_1d(data, support, n_clusters, n_iter=20, weights=None, random_state=42):
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


def wbarycenter_clustering(data, n_clusters, n_iter=20, reg = 1e-1, random_state=42):
    """
    Wasserstein K-Means barycenter-based clustering by Yubo Zhuang et al. (2022)
    for multi-dimensional histograms.

    Parameters:
        data: np.ndarray (n_samples, n_bins)
            Each row is a flattened multi-dimensional histogram (must be normalized).
        n_clusters: int
            Number of clusters.
        n_iter: int
            Number of iterations.
        reg: float
            Entropic regularization for barycenters.
        random_state: int
            Seed for reproducibility.

    Returns:
        assignments: np.ndarray
            Cluster assignments.
        barycenters: list of np.ndarray
            Cluster barycenters.
    """

    rng = np.random.default_rng(random_state)
    n_samples, n_bins = data.shape

    data = data / data.sum(axis=1, keepdims=True)

    assignments = rng.integers(0, n_clusters, size=n_samples)
    for _ in range(n_iter):
        barycenters = []
        for k in range(n_clusters):
            members = data[assignments == k].T  # (n_bins, n_members)
            if members.shape[1] == 0:
                barycenters.append(data[rng.integers(n_samples)])
            else:
                bary = ot_barycenter(members, np.ones(members.shape[1]) / members.shape[1], reg)
                barycenters.append(bary)

        barycenters = np.array(barycenters)
        cost_matrix = ot.utils.dist(np.arange(n_bins).reshape(-1, 1), np.arange(n_bins).reshape(-1, 1)) ** 2

        for i in range(n_samples):
            dists = [
                ot.emd2(data[i], bary, cost_matrix)
                for bary in barycenters
            ]
            assignments[i] = np.argmin(dists)

    return assignments, barycenters
