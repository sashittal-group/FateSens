from abc import ABC, abstractmethod
import os
import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Tuple, Optional
import wot
from enum import Enum


class TransportMap(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def construct_tmap(
        self,
        adata: ad.AnnData,
        tmap_model_path: str,
        day_column_name: str = "time_info",
        days: Optional[List[int]] = None,
    ) -> csr_matrix:
        """
        Construct global transport map across all sequential day intervals.
        
        Parameters
        ----------
        adata: ad.AnnData
            scRNAseq data
        tmap_model_path : str
            Path to the WOT transport map model directory
        day_column_name : str, default "time_info"
            Column name in adata.obs containing day/timepoint information
        days : List[int], optional
            List of days to use. If None, uses all unique days in sorted order
        
        Returns
        -------
        tmap : csr_matrix
            Sparse transport map of shape (n_cells, n_cells) representing
            the composed transport across all time intervals with block structure:
            - Rows represent source cells
            - Columns represent target cells
            - Non-zero blocks exist only for sequential day transitions
        """
        pass


class WotTransportMap(TransportMap):
    """
    Transport map using Waddington OT (WOT) framework.
    
    Constructs a global transport map by composing transport maps across 
    sequential time intervals, mapping local coupling indices to global cell indices.
    
    The resulting sparse transition matrix has a block structure where non-zero entries
    only exist for sequential, forward-time intervals:
    
    For days [2, 4, 6], the block structure is:
    
        Source \\ Target   | Day 4 Cells  | Day 6 Cells
        ------------------|--------------|-------------
        Day 2 Cells        | γ_{2→4}      | 0
        Day 4 Cells        | 0            | γ_{4→6}
    
    where γ_{i→j} represents the OT coupling (transport probabilities) from day i to day j.
    This structure ensures transitions only occur between consecutive timepoints.
    """
    
    def __init__(self):
        super().__init__()
        pass
    
    
    def _get_day_indices(self, day: int, day_column_name: str) -> np.ndarray:
        return np.where(self.adata.obs[day_column_name] == day)[0]
    
    def _normalize_tmap(
        self,
        tmap: csr_matrix,
    )->csr_matrix:
        row_sum = np.array(tmap.sum(axis=1)).flatten()
        row_sum[row_sum == 0] = 1
        tmap_normalized = csr_matrix(tmap.multiply(1 / row_sum[:, None]))
        return tmap_normalized
    
    def construct_tmap(
        self,
        adata: ad.AnnData,
        tmap_model_path: str = "tmaps",
        day_column_name: str = "time_info",
        days: Optional[List[int]] = None,
    ) -> csr_matrix:
        self.adata = adata
        tmap_model_path_wot = os.path.join(tmap_model_path, "serum")
        if not os.path.exists(tmap_model_path):
            os.makedirs(tmap_model_path, exist_ok=True)
            ot_model = wot.ot.OTModel(
                adata, 
                day_field=day_column_name, 
                epsilon=0.05, 
                lambda1=1, 
                lambda2=50, 
                growth_iters=3
            )
            ot_model.compute_all_transport_maps(tmap_out=tmap_model_path_wot)
        tmap_model = wot.tmap.TransportMapModel.from_directory(tmap_model_path_wot)
        
        if days is None:
            days = sorted(self.adata.obs[day_column_name].unique())
        
        global_rows_list = []
        global_cols_list = []
        global_data_list = []
        
        for i in range(len(days) - 1):
            source_day = days[i]
            target_day = days[i + 1]
            
            coupling = tmap_model.get_coupling(source_day, target_day)
            
            coo = csr_matrix(coupling.X).tocoo()
            
            source_indices = np.where(self.adata.obs[day_column_name] == source_day)[0]
            target_indices = np.where(self.adata.obs[day_column_name] == target_day)[0]
            
            global_rows = source_indices[coo.row]
            global_cols = target_indices[coo.col]
            
            global_rows_list.append(global_rows)
            global_cols_list.append(global_cols)
            global_data_list.append(coo.data)
        
        all_rows = np.concatenate(global_rows_list)
        all_cols = np.concatenate(global_cols_list)
        all_data = np.concatenate(global_data_list)
        
        tmap = csr_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(self.adata.n_obs, self.adata.n_obs)
        )
        
        return self._normalize_tmap(tmap)
    

class TransportMapType(Enum):
    WOT = "wot"

def get_transport_map(
    adata: ad.AnnData,
    tmap_model_path: str = "tmaps",
    day_column_name: str = "time_info",
    days: Optional[List[int]] = None,
    type: TransportMapType = TransportMapType.WOT,
) -> TransportMap:
    if type == TransportMapType.WOT:
        tmap = WotTransportMap()

    return tmap.construct_tmap(
        adata,
        tmap_model_path,
        day_column_name,
        days,
    ) 
