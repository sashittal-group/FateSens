from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple, Optional
import anndata as ad
import numpy as np

class NeighborEstimator:
    def get_neighbors(self, data_point, data_set, k):
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(data_set)
        distances, indices = nbrs.kneighbors(data_point)
        return indices[:,1:]
    

def compute_neighbors(adata: ad.AnnData, days_t0: Optional[List[int]] = None, day_column_name: str = "time_info", n_neighbors=200):
    neigbour_estimator = NeighborEstimator()
    if days_t0 is None:
        days_t0 = sorted(adata.obs[day_column_name].unique())[:-1]
    indices_2_4 = adata.obs[day_column_name].isin(days_t0)
    return neigbour_estimator.get_neighbors(
        adata[indices_2_4].obsm["X_emb"],
        adata[indices_2_4].obsm["X_emb"],
        n_neighbors,
    )