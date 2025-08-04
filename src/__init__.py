"""
src package initializer.

This package includes modules for clustering methods and distribution utilities,
primarily used in Wasserstein-based clustering of income distributions.
"""

# Explicit imports for convenience
from .clustering_methods import (
    bary_WKMeans,
    dist_WKMeans
)

from .utils import (
    reconstruct_joint_distribution,
    normalize_supports,
    create_regular_grid,
    project_distribution_on_grid,
    computeDistanceMatrix,
    compute_barycenter_for_cluster,
    plot_projected_distributions
)
