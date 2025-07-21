import numpy as np
import ot  # POT library

from joblib import Parallel, delayed


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


def compute_distance(i, j, data_flat, supports):
    """Compute the Wasserstein distance between distributions i and j using their specific supports."""
    cost_matrix = ot.dist(supports[i], supports[j])
    return ot.emd2(data_flat[i], data_flat[j], cost_matrix)


def computeDistanceMatrix(data, supports, save=False, filepath='../data/Dis_mat.txt'):
    """
    Compute the symmetric pairwise Wasserstein distance matrix between histograms.
    """
    n_samples = data.shape[0]
    n_bins = np.prod(data.shape[1:])

    # Flatten and normalize
    data_flat = data.reshape(n_samples, n_bins)
    data_flat /= data_flat.sum(axis=1, keepdims=True)

    pairwise = np.zeros((n_samples, n_samples))

    # Compute only upper triangle (symmetric matrix)
    results = Parallel(n_jobs=-1)(
        delayed(compute_distance)(i, j, data_flat, supports)
        for i in range(n_samples) for j in range(i+1, n_samples)
    )

    # Fill the matrix
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
