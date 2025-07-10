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
    a = np.array(age_weights, dtype=np.float64)
    a = a / a.sum()  # Normalize to get a probability distribution

    b = np.ones(len(income_values)) / len(income_values)  # Uniform distribution over income deciles

    # Cost matrix: |R_j - m(T_i)|
    C = np.abs(np.array(income_values)[None, :] - np.array(income_median_by_age)[:, None])

    # Solve the discrete optimal transport problem
    pi = ot.emd(a, b, C)

    return pi


def compute_row(data_flat_T, cost_matrix, i):
    return ot.emd2(data_flat_T[:, i], data_flat_T, cost_matrix)


def computeDistanceMatrix(data, save=False, filepath='../data/Dis_mat.txt'):
    """
    Compute the symmetric pairwise Wasserstein distance matrix between histograms.
    """
    n_samples = data.shape[0]
    shape = data.shape[1:]
    n_bins = np.prod(shape)

    # Flatten and normalize
    data_flat = data.reshape(n_samples, n_bins)
    data_flat /= data_flat.sum(axis=1, keepdims=True)
    data_flat_T = data_flat.T

    # Cost matrix
    coords = np.array([(i, j) for i in range(shape[0]) for j in range(shape[1])])
    cost_matrix = ot.dist(coords, coords)

    # Parallel computation of rows
    rows = Parallel(n_jobs=-1)(
        delayed(compute_row)(data_flat_T, cost_matrix, i)
        for i in range(n_samples)
    )
    pairwise = np.vstack(rows)

    # Save if requested
    if save:
        np.savetxt(filepath, pairwise, fmt='%.6e')
        print(f"➡️ Pairwise distance matrix saved to: {filepath}")

    return pairwise
