"""Ridge estimation and detection utilities for cellular trajectory analysis."""

from sklearn.neighbors import radius_neighbors_graph
from scipy.sparse.csgraph import connected_components
from collections import Counter
import numpy as np
from typing import List, Tuple, Optional

try:
    from sconce.EucSCMS import SCMS
except ImportError:
    raise ImportError("sconce package required: pip install sconce")


class RidgeEstimator:
    """Estimates ridge curves using Submanifold Clustering with Manifold Similarity."""

    def __init__(self):
        """Initialize the ridge estimator."""
        pass

    def get_ridge_of_cells(self, coordinates, intensities):
        """
        Calculate ridge points using SCMS algorithm.

        Parameters
        ----------
        coordinates : array-like of shape (n_samples, n_features)
            Spatial coordinates of cells in embedding space.
        intensities : array-like of shape (n_samples,)
            Intensity/sensitivity values for each cell.

        Returns
        -------
        ridge_points : ndarray of shape (n_ridge_points, n_features)
            Coordinates of detected ridge points.
        """
        scms_history = SCMS(
            mesh_0=coordinates,
            data=coordinates,
            d=1,  # d=1 to find 1D ridges (curves)
            h=None,  # Auto-calculate bandwidth or set manually (e.g., 0.5)
            wt=intensities,  # Intensity values to guide ridge detection
            stop_cri='proj_grad'
        )
        return scms_history[:, :, -1]


class RidgeIndices:
    """Identifies connected ridge components and maps them to cell indices."""

    def __init__(self, ridge_estimator: RidgeEstimator = None):
        """
        Initialize ridge indices detector.

        Parameters
        ----------
        ridge_estimator : RidgeEstimator, optional
            Ridge estimator instance. If None, creates new instance.
        """
        self.ridge_estimator = ridge_estimator or RidgeEstimator()

    def get_connected_components(self, ridge_points, radius=100):
        """
        Identify connected components in ridge points.

        Parameters
        ----------
        ridge_points : ndarray of shape (n_ridge_points, n_features)
            Coordinates of ridge points.
        radius : float, default=100
            Radius for neighborhood connectivity (tune to embedding scale).

        Returns
        -------
        labels : ndarray
            Component labels for each ridge point.
        """
        G = radius_neighbors_graph(
            ridge_points,
            radius=radius,
            mode='connectivity',
            include_self=False
        )
        n_components, labels = connected_components(G)
        return labels

    def get_k_largest_component(self, labels, k=2):
        """
        Get k largest connected components by point count.

        Parameters
        ----------
        labels : array-like
            Component labels from get_connected_components.
        k : int, default=2
            Number of largest components to return.

        Returns
        -------
        largest_k_labels : list
            Labels of k largest components, sorted by size.
        """
        label_counts = Counter(labels)
        sorted_labels = [label for label, count in label_counts.most_common()]
        largest_k_labels = sorted_labels[:k]
        return largest_k_labels

    def get_k_ridge_points_on_adata(
        self,
        adata,
        ridge_points,
        labels,
        coordinates,
        n_components=2
    ):
        """
        Map ridge points to nearest cells in AnnData object.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix with observations.
        ridge_points : ndarray of shape (n_ridge_points, n_features)
            Coordinates of ridge points.
        labels : ndarray
            Component labels for ridge points.
        coordinates : ndarray of shape (n_cells, n_features)
            Coordinates of cells (e.g., from adata.obsm['X_emb']).
        n_components : int, default=2
            Number of ridge components to extract.

        Returns
        -------
        ridge_segments_adata_obs_index : list of ndarray
            Cell observation indices for each ridge segment.
        """
        largest_k_labels = self.get_k_largest_component(labels, k=n_components)
        top_n_coordinates_labels = [ridge_points[labels == label] for label in largest_k_labels]

        ridge_segments_adata_obs_index = []
        for ridge_segment in top_n_coordinates_labels:
            indices = []
            for point in ridge_segment:
                # Find nearest cell in coordinates
                dists = np.linalg.norm(coordinates - point, axis=1)
                nearest_idx = np.argmin(dists)
                indices.append(adata.obs.index[nearest_idx])
            indices = np.unique(indices)
            ridge_segments_adata_obs_index.append(indices)

        return ridge_segments_adata_obs_index

    def get_ridge_indices(
        self,
        adata,
        sensitivities,
        coordinates,
        n_components=2,
        radius=100
    ):
        """
        Full pipeline: estimate ridge, find components, and map to cell indices.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix.
        sensitivities : array-like of shape (n_cells,)
            Sensitivity/intensity values for each cell.
        coordinates : ndarray of shape (n_cells, n_features)
            Cell coordinates in embedding space.
        n_components : int, default=2
            Number of ridge components to extract.
        radius : float, default=100
            Radius for connectivity (tune to embedding scale).

        Returns
        -------
        ridge_segments_adata_obs_index : list of ndarray
            Cell observation indices for each detected ridge segment.
        """
        intensities = sensitivities
        ridge_points = self.ridge_estimator.get_ridge_of_cells(coordinates, intensities)
        labels = self.get_connected_components(ridge_points, radius=radius)
        ridge_segments_adata_obs_index = self.get_k_ridge_points_on_adata(
            adata, ridge_points, labels, coordinates, n_components=n_components
        )
        return ridge_segments_adata_obs_index


def estimate_ridge(
    adata,
    sensitivities,
    n_components=2,
    radius=100,
    ridge_estimator=None,
    use_rep = "X_emb",
    day_t0: List[int] = [2, 4],
    day_column_name: str = "time_info",
):
    """
    Estimate ridge in cell trajectory and map to cell indices.

    Standalone function to detect ridge curves in embedding space and identify
    cells closest to the ridge segments.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    sensitivities : array-like of shape (n_cells,)
        Sensitivity/intensity values for each cell (e.g., fate probabilities).
    coordinates : ndarray of shape (n_cells, n_features)
        Cell coordinates in embedding space (e.g., from adata.obsm['X_emb']).
    n_components : int, default=2
        Number of ridge components to extract.
    radius : float, default=100
        Radius for connectivity in ridge component detection (tune to embedding scale).
    ridge_estimator : RidgeEstimator, optional
        Custom ridge estimator. If None, creates new instance.

    Returns
    -------
    ridge_segments : list of ndarray
        Cell observation indices for each detected ridge segment.

    Examples
    --------
    >>> ridge_segments = estimate_ridge(
    ...     adata,
    ...     sensitivities=adata.obs['fate_probability'].values,
    ...     coordinates=adata.obsm['X_emb'],
    ...     n_components=2,
    ...     radius=100
    ... )
    >>> for i, segment in enumerate(ridge_segments):
    ...     print(f"Ridge segment {i}: {len(segment)} cells")
    """
    if ridge_estimator is None:
        ridge_estimator = RidgeEstimator()

    ridge_indices = RidgeIndices(ridge_estimator=ridge_estimator)
    adata_subset = adata[adata.obs[day_column_name].isin(day_t0)]
    coordinates = adata_subset.obsm[use_rep]
    ridge_segments = ridge_indices.get_ridge_indices(
        adata=adata_subset,
        sensitivities=sensitivities,
        coordinates=coordinates,
        n_components=n_components,
        radius=radius
    )
    return ridge_segments
