import numpy as np
import ot  # POT library

from joblib import Parallel, delayed

import matplotlib.pyplot as plt


def reconstruct_joint_distribution_ot(age_weights, income_values, income_median_by_age):
    """
    Reconstruct the joint distribution between age groups and income levels for a given IRIS.

    Parameters:
        age_weights : Population counts for each age group T_i. ex : (A_1, ..., A_n)
        income_values : Income values corresponding to the deciles. ex : (D_1, ..., D_9)
        income_median_by_age : National median income for each age group. ex : (m(T_1), ..., m(T_n))

    Returns:
        np.ndarray: Optimal transport plan pi,
            (its a discrete approxi of the joint distribution between age and income)
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


def create_regular_grid(n_bins_age=7, n_bins_income=9):
    grid_age = np.linspace(0, 1, n_bins_age)
    grid_income = np.linspace(0, 1, n_bins_income)

    grid_x, grid_y = np.meshgrid(grid_age, grid_income, indexing='ij')
    grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # shape (n_bins, 2)
    return grid


def project_distribution_on_grid(support, distribution, grid):
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
    Compute the symmetric pairwise Wasserstein distance matrix between histograms
    projected on a common grid.

    Parameters:
        data: np.ndarray of shape (n_samples, n_bins)
            Projected histograms (e.g. projected_distributions).
        grid: np.ndarray of shape (n_bins, 2)
            Common support grid.
        save: bool
            Whether to save the matrix.
        filepath: str
            Path for saving the matrix (if save=True).
    """
    n_samples = data.shape[0]

    # Normalize histograms
    data_flat = data / data.sum(axis=1, keepdims=True)

    # Compute cost matrix once for the common grid
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
    rng_local = np.random.default_rng(seed)
    indices_k = np.where(assignments == k)[0]
    if len(indices_k) == 0:
        return data[rng_local.integers(0, len(data))]
    else:
        cluster_hists = data[indices_k].T
        return ot.bregman.barycenter(cluster_hists, cost_matrix, reg)


# Plot

def plot_projected_distribution(distribution, grid_shape, support_min, support_max, title="Distribution projetée"):
    """
    distribution : array de shape (n_bins_age * n_bins_income,)
    grid_shape : (n_bins_age, n_bins_income)
    support_min, support_max : array(2,) pour dénormaliser les axes
    """
    n_bins_age, n_bins_income = grid_shape

    # Reshape en matrice
    matrix = distribution.reshape(grid_shape)

    # Définir les bornes en âge et revenu dénormalisés
    age_min, income_min = support_min
    age_max, income_max = support_max

    extent = [income_min, income_max, age_min, age_max]  # [xmin, xmax, ymin, ymax]

    plt.imshow(matrix, origin='lower', cmap='viridis', extent=extent, aspect='auto')
    plt.title(title)
    plt.xlabel("Revenu (échelle originale)")
    plt.ylabel("Âge (échelle originale)")
    plt.colorbar(label="Poids")
    plt.grid(False)
    plt.show()
