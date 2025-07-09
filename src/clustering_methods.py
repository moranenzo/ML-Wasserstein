import numpy as np
import ot
from tqdm.notebook import tqdm


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


def wbarycenter_clustering(data, n_clusters=3, n_iter=10, reg=1e-1, random_state=None):
    """
    Wasserstein K-Means barycenter-based clustering by Yubo Zhuang et al. (2022)
    for n-dimensional histograms.

    Parameters:
        data: np.ndarray of shape (n_samples, shape)
            Input histograms
        n_clusters: int
            Number of clusters.
        n_iter: int
            Number of iterations.
        reg: float
            Entropic regularization strength.
        random_state: int or None
            Random seed for reproducibility.

    Returns:
        assignments: np.ndarray of shape (n_samples,)
            Cluster assignments.
        barycenters: list of np.ndarray
            Cluster barycenters.
    """
    rng = np.random.default_rng(random_state)
    n_samples = data.shape[0]
    shape = data.shape[1:]
    n_bins = np.prod(shape)

    # Flatten the input distributions
    data_flat = data.reshape(n_samples, n_bins) 
    # Optional: uncomment below (if needed) to avoid numerical issues with NaN or empty barycenters
    # data_flat = np.where(data_flat < 1e-12, 1e-12, data_flat)

    # Normalize distributions to ensure they are valid probabilities
    data_flat /= data_flat.sum(axis=1, keepdims=True)

    # Initialization: randomly choose initial barycenters among the data
    bary_flat = data_flat[rng.choice(n_samples, size=n_clusters, replace=False)]

    # Compute cost matrix
    coords = np.array([(i, j) for i in range(shape[0]) for j in range(shape[1])])
    cost_matrix = ot.dist(coords, coords)

    for it in tqdm(range(n_iter), desc="Wasserstein clustering iterations"):

        # Step 1: Assignment - assign each distribution to the closest barycenter
        assignments = np.zeros(n_samples, dtype=int)

        for i in tqdm(range(n_samples), desc="Assigning to barycenters"):
            distances = np.zeros(n_clusters, dtype=float)
            for j in range(n_clusters):
                distances[j] = ot.emd2(data_flat[i], bary_flat[j], cost_matrix)
            assignments[i] = np.argmin(distances)

        # Step 2: Update - recompute the barycenters
        bary_flat = []
        for k in tqdm(range(n_clusters), desc="Recomputing barycenters"):
            # Select all members of cluster k
            members = data_flat[assignments == k].T

            # If the cluster is empty, reinitialize with a random sample
            if len(members) == 0:
                idx = rng.integers(0, data.shape[0])
                bary_flat.append(data_flat[idx])
                continue

            # Compute regularized Wasserstein barycenter
            bary_flat.append(ot.bregman.barycenter(members, cost_matrix, reg))

    # Reshape barycenters
    barycenters = [b.reshape(shape) for b in bary_flat]

    return assignments, barycenters


# K-Means clustering pairwise-based from Zhuang et al. (2022)

def wpairwise_clustering(data, n_clusters=2, n_iter=20, random_state=None):
    rng = np.random.default_rng(random_state)
    n_samples = data.shape[0]
    shape = data.shape[1:]
    n_bins = np.prod(shape)

    barycenters = data[rng.choice(n_samples, size=n_clusters, replace=False)]

    # Flatten distributions and barycenters
    data_flat = data.reshape(n_samples, n_bins) 
    # if NaN or empty barycenters, add the line below :
    # data_flat = np.where(data_flat < 1e-12, 1e-12, data_flat)

    # Normalize to have probabilities
    data_flat /= data_flat.sum(axis=1, keepdims=True)

    # Compute cost matrix
    coords = np.array([(i, j) for i in range(shape[0]) for j in range(shape[1])])
    cost_matrix = ot.dist(coords, coords)

    # Compute full pairwise distance matrix (symmetric)
    pairwise_w2 = np.zeros((n_samples, n_samples))
    for i in tqdm(range(n_samples), desc="Computing pairwise distance matrix..."):
        for j in range(i, n_samples):
            d = ot.emd2(data_flat[i], data_flat[j], cost_matrix)
            pairwise_w2[i, j] = d
            pairwise_w2[j, i] = d

    # Initialize assignments randomly
    assignments = rng.integers(0, n_clusters, size=n_samples)

    for it in tqdm(range(n_iter), desc="Wasserstein pairwise clustering iterations"):
        new_assignments = np.zeros_like(assignments)

        for i in range(n_samples):
            avg_dists = np.zeros(n_clusters)
            for k in range(n_clusters):
                members = np.where(assignments == k)[0]
                if len(members) > 0:
                    avg_dists[k] = pairwise_w2[i, members].mean()
                else:
                    avg_dists[k] = np.inf
            new_assignments[i] = np.argmin(avg_dists)

        if np.array_equal(assignments, new_assignments):
            print(f"Converged at iteration {it}")
            break

        assignments = new_assignments

    return assignments