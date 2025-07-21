import numpy as np
import ot
from tqdm.notebook import tqdm

from utils import computeDistanceMatrix


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


def bary_WKMeans(data, supports, n_clusters=3, n_iter=10, reg=1e-1, random_state=None):
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

    # Flatten and Normalize the input distributions
    data_flat = data.reshape(n_samples, n_bins)
    data_flat /= data_flat.sum(axis=1, keepdims=True)

    # Initialization: randomly choose initial barycenters among the data
    indices = rng.choice(n_samples, size=n_clusters, replace=False)
    bary_flat = data_flat[indices]
    bary_supports = supports[indices]

    for it in tqdm(range(n_iter), desc="Wasserstein clustering iterations"):

        # Step 1: Assignment - assign each distribution to the closest barycenter
        assignments = np.zeros(n_samples, dtype=int)

        for i in tqdm(range(n_samples), desc="Assigning to barycenters"):
            distances = np.zeros(n_clusters, dtype=float)
            for j in range(n_clusters):
                # Compute cost matrix
                cost_matrix = ot.dist(supports[i], supports[j])
                distances[j] = ot.emd2(data_flat[i], bary_flat[j], cost_matrix)
            assignments[i] = np.argmin(distances)

        # Step 2: Update - recompute the barycenters
        bary_flat = []
        bary_supports = []

        for k in tqdm(range(n_clusters), desc="Recomputing barycenters"):
            # Indices des distributions du cluster k
            cluster_indices = np.where(assignments == k)[0]

            # If the cluster is empty, reinitialize with a random sample
            if len(cluster_indices) == 0:
                idx = rng.integers(0, data.shape[0])
                bary_flat.append(data_flat[idx])
                bary_supports.append(supports[idx])
                continue

            # Récupérer les histogrammes et les supports associés
            cluster_hists = [data_flat[i] for i in cluster_indices]
            cluster_supports = [supports[i] for i in cluster_indices]

            # Construire le support commun (union des supports)
            combined_support = np.unique(np.vstack(cluster_supports), axis=0)

            # Reprojeter chaque histogramme sur le support commun
            projected_hists = []
            for hist, supp in zip(cluster_hists, cluster_supports):
                proj = np.zeros(len(combined_support))
                for i, atom in enumerate(supp):
                    # Trouver l’indice correspondant dans le support commun
                    idx = np.where((combined_support == atom).all(axis=1))[0][0]
                    proj[idx] = hist[i]
                projected_hists.append(proj)

            projected_hists = np.array(projected_hists).T  # (n_bins, n_distributions)

            # Recalculer la cost matrix sur le support commun
            cost_matrix = ot.dist(combined_support, combined_support)

            # Calcul du barycentre régularisé
            bary = ot.bregman.barycenter(projected_hists, cost_matrix, reg)

            # Stocker le barycentre et son support
            bary_flat.append(bary)
            bary_supports.append(combined_support)

    # Reshape barycenters
    bary_hists = [b.reshape(shape) for b in bary_flat]

    return assignments, bary_hists, bary_supports


# D-WKM : K-Means clustering pairwise-based from Zhuang et al. (2022)

def dist_WKMeans(data, supports, dist_matrix=None, n_clusters=2, n_iter=10, random_state=None):
    rng = np.random.default_rng(random_state)
    n_samples = data.shape[0]
    shape = data.shape[1:]
    n_bins = np.prod(shape)

    # Flatten and Normalize the input distributions
    data_flat = data.reshape(n_samples, n_bins)
    data_flat /= data_flat.sum(axis=1, keepdims=True)

    if dist_matrix is None:
        dist_matrix = computeDistanceMatrix(data, supports)

    # Initialize assignments randomly
    assignments = rng.integers(0, n_clusters, size=n_samples)

    for it in tqdm(range(n_iter), desc="Wasserstein pairwise clustering iterations"):
        new_assignments = np.zeros_like(assignments)

        for i in range(n_samples):
            avg_dists = np.zeros(n_clusters)
            for k in range(n_clusters):
                members = np.where(assignments == k)[0]
                if len(members) > 0:
                    avg_dists[k] = dist_matrix[i, members].mean()
                else:
                    avg_dists[k] = np.inf
            new_assignments[i] = np.argmin(avg_dists)

        if np.array_equal(assignments, new_assignments):
            print(f"Converged at iteration {it}")
            break

        assignments = new_assignments

    return assignments


# W-SDP : Semidefinite Program relaxation of the distance-based K-means
