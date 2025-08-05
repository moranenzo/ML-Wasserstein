import numpy as np
import ot
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

"""
utils.py

Utility functions for joint distribution reconstruction, projection,
distance matrix computation, barycenter calculation, and visualization
in the context of age-income distributions using optimal transport.
"""


def reconstruct_joint_distribution(age_weights, income_values, income_median_by_age):
    """
    Reconstruct the joint distribution between age groups and income deciles
    via discrete optimal transport using |income - median_income| cost.

    Parameters:
        age_weights (array-like): Population counts by age group T_i. ex : (A_1, ..., A_n)
        income_values (array-like): Values corresponding to the income deciles. ex : (D_1, ..., D_9)
        income_median_by_age (array-like): National median income by age group. ex : (m(T_1), ..., m(T_n))

    Returns:
        pi (np.ndarray): Optimal transport plan (discrete approximation of the joint distribution between age and income)
        support (np.ndarray): Support coordinates as (age_index, income_value)
    """
    age_weights = np.array(age_weights, dtype=np.float64)
    income_values = np.array(income_values, dtype=np.float64)
    income_median_by_age = np.array(income_median_by_age, dtype=np.float64)

    a = age_weights / age_weights.sum()  # Normalize to get a probability distribution
    b = np.ones(len(income_values)) / len(income_values)  # Uniform distribution over income deciles

    # Cost matrix: |R_j - m(T_i)|
    C = np.abs(income_values[None, :] - income_median_by_age[:, None])

    # Solve the discrete optimal transport problem
    pi = ot.emd(a, b, C)

    # Construct the support: all (age_index, income_value) combinations
    support = np.array([
        (age_idx, income_val)
        for age_idx in range(len(age_weights))
        for income_val in income_values
    ])

    return pi, support


def normalize_supports(supports):
    """
    Normalize multiple supports

    Parameters:
        supports (list of np.ndarray): List of d-dimensional points (arrays).

    Returns:
        normalized (list of np.ndarray): List of normalized supports.
        min_vals (np.ndarray): Minimum values across all supports for each dimension.
        max_vals (np.ndarray): Maximum values across all supports for each dimension.
    """
    all_supports = np.vstack(supports)
    min_vals = all_supports.min(axis=0)
    max_vals = all_supports.max(axis=0)
    normalized = [(s - min_vals) / (max_vals - min_vals) for s in supports]
    return normalized, min_vals, max_vals


def create_regular_grid(n_bins_age=7, n_bins_income=9):
    """
    Create a regular 2D grid on the unit square [0,1] x [0,1].

    Parameters:
        n_bins_age (int): Number of bins in the age dimension.
        n_bins_income (int): Number of bins in the income dimension.

    Returns:
        grid (np.ndarray): Array of shape (n_bins_age * n_bins_income, 2) containing grid coordinates.

    """
    grid_age = np.linspace(0, 1, n_bins_age)
    grid_income = np.linspace(0, 1, n_bins_income)

    grid_x, grid_y = np.meshgrid(grid_age, grid_income, indexing='ij')
    grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # shape (n_bins, 2)
    return grid


def project_distribution_on_grid(support, distribution, grid):
    """
    Project a discrete distribution defined on a support onto a fixed regular
    grid by assigning the mass of each support point to the closest grid point.

    Parameters:
        support (np.ndarray): Support points of the distribution (shape: n_points x 2).
        distribution (np.ndarray): Weights associated with each support point.
        grid (np.ndarray): Coordinates of the grid points (shape: n_bins x 2).

    Returns:
        projected (np.ndarray): Histogram of the projected distribution on the grid (length = n_bins).
    """
    projected = np.zeros(len(grid))
    for i in range(len(support)):
        s = support[i]
        mass = float(distribution[i])  # <-- cast explicite

        distances = np.linalg.norm(grid - s, axis=1)
        closest_idx = np.argmin(distances)
        projected[closest_idx] += mass

    return projected


def computeDistanceMatrix(data, grid, save=False, filepath='../data/Dis_mat.txt'):
    """
    Compute the symmetric pairwise Wasserstein squared distance matrix between histograms
    projected on a common grid.

    Parameters:
        data (np.ndarray of shape (n_samples, n_bins)): Projected histograms.
        grid (np.ndarray of shape (n_bins, 2)): Common support grid.
        save (bool): Whether to save the matrix.
        filepath (str): Path for saving the matrix (if save=True).
    
    Returns:
        pairwise (np.ndarray): Symmetric distance matrix of shape (n_samples, n_samples).
    """
    n_samples = data.shape[0]
    # Normalize histograms to sum to 1
    data_flat = data / data.sum(axis=1, keepdims=True)
    # Compute cost matrix between grid points
    cost_matrix = ot.dist(grid, grid)

    pairwise = np.zeros((n_samples, n_samples))

    # Parallelized computation
    results = Parallel(n_jobs=-1)(
        delayed(ot.emd2)(data_flat[i], data_flat[j], cost_matrix)
        for i in range(n_samples) for j in range(i+1, n_samples)
    )

    # Fill the full matrix
    k = 0
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            pairwise[i, j] = results[k]
            pairwise[j, i] = results[k]
            k += 1

    if save:
        np.savetxt(filepath, pairwise, fmt='%.6e')
        print(f"Pairwise distance matrix saved to: {filepath}")

    return pairwise


def compute_barycenter_for_cluster(k, assignments, data, cost_matrix, reg, seed):
    """
    Compute the entropically regularized Wasserstein barycenter of
    distributions assigned to cluster k.

    If the cluster is empty, returns a random distribution from data.

    Parameters
        k (int): Index of the cluster.
        assignments (np.ndarray): Current cluster assignments (length = n_samples).
        data (np.ndarray): Histograms of all samples (shape: n_samples x n_bins).
        cost_matrix (np.ndarray): Cost matrix between grid points.
        reg (float): Entropic regularization parameter.
        seed (int): Random seed for reproducibility.

    Returns:
        barycenter (np.ndarray): Computed barycenter histogram for cluster k.
    """
    rng = np.random.default_rng(seed)
    indices = np.where(assignments == k)[0]

    if len(indices) == 0:
        # Return a random histogram if cluster is empty
        return data[rng.integers(0, len(data))]
    else:
        cluster_hists = data[indices].T
        barycenter = ot.bregman.barycenter(cluster_hists, cost_matrix, reg)
        return barycenter


def plot_projected_distributions(distributions, grid_shape, support_min, support_max, title="Projected Distribution"):
    """
    Visualize one or more 2D projected distributions as images, with axes
    rescaled to original units.

    Parameters:
        distributions (np.ndarray or list of np.ndarray): Single histogram or list of histograms (flattened).
        grid_shape (tuple): Grid shape (n_bins_age, n_bins_income).
        support_min (np.ndarray): Min values per dimension (for axis rescaling).
        support_max (np.ndarray): Max values per dimension.
        title (str): Base title for plots.

    Returns:
        None
    """
    if isinstance(distributions, np.ndarray):
        distributions = [distributions]

    n = len(distributions)

    # Determine layout
    if n == 1:
        n_rows, n_cols = 1, 1
    elif n == 2:
        n_rows, n_cols = 1, 2
    elif n == 4:
        n_rows, n_cols = 2, 2
    else:
        n_cols = 3
        n_rows = int(np.ceil(n / 3))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = np.array(axs).reshape(-1)  # Flatten in case it's 2D

    for i, dist in enumerate(distributions):
        ax = axs[i]
        matrix = dist.reshape(grid_shape)
        extent = [support_min[1], support_max[1], support_min[0], support_max[0]]

        im = ax.imshow(matrix, origin='lower', cmap='viridis', extent=extent, aspect='auto')
        ax.set_title(f"{title} {i}")
        ax.set_xlabel("Income (original scale)")
        ax.set_ylabel("Age (original scale)")
        fig.colorbar(im, ax=ax, orientation='vertical', label="Mass")

    # Hide any unused axes
    for j in range(len(distributions), len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()
