import numpy as np
import ot  # POT library


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
