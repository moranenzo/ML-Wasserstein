import numpy as np
import ot
from ot.bregman import barycenter as ot_barycenter
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


def create_nd_cost_matrix(shape):
    """
    Create a cost matrix for an n-dimensional grid of shape `shape`.
    Cost is the squared Euclidean distance between bin coordinates.
    """
    grid = np.indices(shape).reshape(len(shape), -1).T  # shape (n_bins, n_dims)
    return ot.utils.dist(grid, grid) ** 2  # shape (n_bins, n_bins)


def wbarycenter_clustering_nd(data, n_clusters, n_iter=20, reg=1e-1, random_state=42, debug=False):
    """
    Wasserstein barycenter clustering for n-dimensional histograms.

    Parameters:
        data: np.ndarray of shape (n_samples, *shape)
            Input histograms (e.g., 2D: (n_samples, n_age, n_decile))
        n_clusters: int
            Number of clusters.
        n_iter: int
            Number of iterations.
        reg: float
            Entropic regularization strength.
        random_state: int
            For reproducibility.
        debug: bool
            If True, prints debug information (NaNs, zero rows, sums, etc.)

    Returns:
        assignments: np.ndarray of shape (n_samples,)
            Cluster assignments.
        barycenters: list of np.ndarray of shape `shape`
            Cluster barycenters.
    """
    rng = np.random.default_rng(random_state)
    n_samples = data.shape[0]
    shape = data.shape[1:]
    n_bins = np.prod(shape)

    # Flatten and normalize each histogram
    data_flat = data.reshape(n_samples, n_bins)
    data_flat = np.where(data_flat < 1e-12, 1e-12, data_flat)
    row_sums = data_flat.sum(axis=1, keepdims=True)
    zero_rows = np.where(row_sums.squeeze() == 0)[0]
    if debug and len(zero_rows) > 0:
        print(f"[DEBUG] Warning: {len(zero_rows)} zero-sum rows in input data.")
    data_flat = data_flat / row_sums

    if debug:
        print(f"[DEBUG] data_flat contains NaN: {np.isnan(data_flat).any()}")
        print(f"[DEBUG] data_flat contains inf: {np.isinf(data_flat).any()}")
        print(f"[DEBUG] data_flat min: {data_flat.min()}, max: {data_flat.max()}")

    # Initialize cluster assignments randomly
    assignments = rng.integers(0, n_clusters, size=n_samples)

    # Compute cost matrix
    cost_matrix = create_nd_cost_matrix(shape)

    for it in tqdm(range(n_iter), desc="Wasserstein clustering iterations"):
        barycenters_flat = []
        for k in range(n_clusters):
            members = data_flat[assignments == k].T  # (n_bins, n_members)
            if members.shape[1] == 0:
                if debug:
                    print(f"[DEBUG] No members in cluster {k}, reinitializing randomly.")
                bary = data_flat[rng.integers(n_samples)]
            else:
                bary = ot.bregman.barycenter(
                    members,
                    M=cost_matrix,
                    reg=reg,
                    numItermax=3000
                )
            if debug:
                if not np.all(np.isfinite(bary)):
                    print(f"[DEBUG] Non-finite barycenter for cluster {k} at iter {it}")
                if bary.sum() == 0:
                    print(f"[DEBUG] Zero barycenter for cluster {k} at iter {it}")
            barycenters_flat.append(bary)

        barycenters_flat = np.array(barycenters_flat)

        # Reassign samples
        for i in tqdm(range(n_samples), desc="Reassigning clusters", leave=False):
            pi = data_flat[i]
            pi_sum = pi.sum()
            if pi_sum == 0 or not np.all(np.isfinite(pi)):
                if debug:
                    print(f"[DEBUG] Sample {i} has invalid pi: sum={pi_sum}, finite={np.all(np.isfinite(pi))}")
                continue
            pi = pi / pi_sum

            dists = []
            for j, bary in enumerate(barycenters_flat):
                bary_sum = bary.sum()
                if bary_sum == 0 or not np.all(np.isfinite(bary)):
                    if debug:
                        print(f"[DEBUG] Cluster {j} barycenter invalid during reassignment.")
                    dists.append(np.inf)
                    continue
                bary_norm = bary / bary_sum

                try:
                    dist = ot.emd2(pi, bary_norm, cost_matrix)
                except AssertionError as e:
                    if debug:
                        print(f"[DEBUG] EMD2 failed at sample {i}, cluster {j}: {str(e)}")
                        print(f"[DEBUG] pi.sum() = {pi.sum()}, bary_norm.sum() = {bary_norm.sum()}")
                    dist = np.inf
                dists.append(dist)

            assignments[i] = np.argmin(dists)

    # Reshape barycenters
    barycenters = [b.reshape(shape) for b in barycenters_flat]

    return assignments, barycenters


"""
def wasserstein_kmeans(data, n_clusters=3, n_iter=10, random_state=42):
    rng = np.random.default_rng(random_state)
    n_samples = data.shape[0]
    shape = data.shape[1:]
    n_bins = np.prod(shape)

    # Normalization
    data = np.where(data < 1e-12, 1e-12, data)
    data = data / data.sum(axis=(1, 2), keepdims=True)

    M = create_nd_cost_matrix(shape)

    # Initialize barycenters with random samples
    barycenters = data[rng.choice(n_samples, size=n_clusters, replace=False)]
    assignments = np.zeros(n_samples, dtype=int)

    for it in tqdm(range(n_iter), desc="Wasserstein k-means"):
        # Assign each sample to the closest barycenter
        for i in range(n_samples):
            dists = [
                ot.emd2(data[i].ravel(), bary.ravel(), M)
                for bary in barycenters
            ]
            assignments[i] = np.argmin(dists)

        # Update barycenters
        new_barycenters = []
        for k in range(n_clusters):
            members = data[assignments == k]
            if len(members) == 0:
                # Empty cluster: reinitialize with a random sample
                new_barycenters.append(data[rng.integers(n_samples)])
            else:
                A = members.reshape(len(members), n_bins).T  # shape (n_bins, n_members)
                bary = ot.barycenter(A, M)
                new_barycenters.append(bary.reshape(shape))

        barycenters = new_barycenters

    return assignments, barycenters
"""
