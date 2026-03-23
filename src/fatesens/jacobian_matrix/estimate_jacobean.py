from ._calculate_neighbor import compute_neighbors
from typing import List, Tuple, Optional
from ..common_utils import parallelize_function

from scipy import sparse
from scipy.sparse import csr_matrix
import numpy as np
from enum import Enum

class JacobianType(Enum):
    FLOW_MAP = "flow_map"
    FATE_PROBABILITY = "fate_probability"


class JacobeanEstimator:
    def get_jacobian(self, neigbour_cells_initial_density, neighbour_cells_final_density, type: JacobianType = JacobianType.FLOW_MAP, beta=10):
        assert neigbour_cells_initial_density.shape[0] == neighbour_cells_final_density.shape[0], "Number of neighbors must match between initial and final densities."

        neigbour_cells_initial_density = neigbour_cells_initial_density.toarray() if sparse.issparse(neigbour_cells_initial_density) else neigbour_cells_initial_density
        neighbour_cells_final_density = neighbour_cells_final_density.toarray() if sparse.issparse(neighbour_cells_final_density) else neighbour_cells_final_density

        X = neigbour_cells_initial_density[:,None] - neigbour_cells_initial_density
        Y = neighbour_cells_final_density[:,None] - neighbour_cells_final_density
        X = X.reshape(-1,X.shape[-1])
        Y = Y.reshape(-1,Y.shape[-1])

        if sparse.issparse(X):
            X = X.toarray()
        if sparse.issparse(Y):
            Y = Y.toarray()

        X = X.T
        Y = Y.T

        M = X.shape[1]
        d = Y.shape[0]
        G = X.shape[0]
        
        if type == JacobianType.FATE_PROBABILITY:
            jacobian_dense = (np.eye(d) - (1.0 / d) * np.ones((d, d))) @ (Y@X.T) @ np.linalg.inv(X @ X.T + beta*M*np.eye(G))
        elif type == JacobianType.FLOW_MAP:
            jacobian_dense = (Y@X.T) @ np.linalg.inv(X @ X.T + beta*M*np.eye(G))
        jacobian_dense = np.round(jacobian_dense, decimals=10)

        jacobian = sparse.csr_matrix(jacobian_dense)
        return jacobian
    
class SensitivityEstimator:
    def estimate(self, jacobian, x_t, Positives, Negatives):
        A = np.zeros((x_t.shape[0], x_t.shape[0]))
        for pos in Positives:
            A += np.dot(pos.T, pos) / (np.linalg.norm(pos @ x_t) + 1e-10)
        for neg in Negatives:
            A -= np.dot(neg.T, neg) / (np.linalg.norm(neg @ x_t) + 1e-10)
        return 2 * jacobian.T @ A @ x_t

class SingularValueEstimator:
    def get_largest_singular_value(self, jacobian):
        try:
            u, s, _ = np.linalg.svd(jacobian.toarray())
            return s[0]
        except:
            return 0

def estimate_jacobian_of_flow_map(adata, x_0: csr_matrix, x_t: csr_matrix, days_t0: Optional[List[int]] = None, day_column_name: str = "time_info", n_neighbors=200, beta=10):
    neighbors = compute_neighbors(
        adata,
        days_t0=days_t0,
        day_column_name=day_column_name,
        n_neighbors=n_neighbors,
    )

    indices_day_t0 = adata.obs[day_column_name].isin(days_t0)
    adata_t0 = adata[indices_day_t0]
    jacobean_estimator = JacobeanEstimator()
    args_ = [
        [x_0[neighbors[i]], x_t[neighbors[i]], JacobianType.FLOW_MAP, beta]
        for i in range(adata_t0.shape[0])
    ]

    all_jacobians = parallelize_function(jacobean_estimator.get_jacobian, args_)
    jacobian = all_jacobians
    return jacobian

def estimate_jacobian_of_fate_probability(adata, x_0: csr_matrix, x_t_probability: csr_matrix, days_t0: Optional[List[int]] = None, day_column_name: str = "time_info", n_neighbors=200, beta=10):
    neighbors = compute_neighbors(
        adata,
        days_t0=days_t0,
        day_column_name=day_column_name,
        n_neighbors=n_neighbors,
    )

    indices_day_t0 = adata.obs[day_column_name].isin(days_t0)
    adata_t0 = adata[indices_day_t0]
    jacobean_estimator = JacobeanEstimator()
    args_ = [
        [x_0[neighbors[i]], x_t_probability[neighbors[i]], JacobianType.FATE_PROBABILITY, beta]
        for i in range(adata_t0.shape[0])
    ]

    all_jacobians = parallelize_function(jacobean_estimator.get_jacobian, args_)
    jacobian = all_jacobians
    return jacobian

def estimate_sensitivity(adata, all_jacobians, x_t, positives: np.array, negatives: np.array, days_t0: Optional[List[int]] = None, day_column_name: str = "time_info"):
    sensitivity_estimator = SensitivityEstimator()
    indices_2_4 = adata.obs[day_column_name].isin(days_t0)
    args_ = [
        [
        all_jacobians[cell_idx],
        x_t[cell_idx].T,
        positives,
        negatives,
        ]
        for cell_idx in range(len(adata[indices_2_4].obs_names))
    ]
    sensitivity = parallelize_function(sensitivity_estimator.estimate, args_)
    return np.array(sensitivity)

def compute_largest_singular_values(jacobians):
    """
    Compute the largest singular value for each jacobian matrix in parallel.
    
    Parameters
    ----------
    jacobians : list
        List of jacobian matrices (sparse csr_matrix format).
    
    Returns
    -------
    np.ndarray
        Array of largest singular values.
    """
    singular_value_estimator = SingularValueEstimator()
    args_ = [[jacobian] for jacobian in jacobians]
    sing_vals = parallelize_function(singular_value_estimator.get_largest_singular_value, args_)
    return 2**(np.log10(np.array(sing_vals)+10e-10))
