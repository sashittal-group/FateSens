from scipy.sparse import csr_matrix
from typing import List, Tuple, Optional
import anndata as ad
import numpy as np


class FlowMap:
    """
    Compute flow maps (transported gene expressions) using transport maps.
    
    A flow map represents the predicted gene expression of source cells
    after transport through a transport map to target cells.
    """
    
    def __init__(self):
        """Initialize FlowMap."""
        pass
    
    def day_t0_expression(
        self,
        adata: ad.AnnData,
        days_t0: Optional[List[int]] = None,
        day_column_name: str = "time_info",
    ) -> csr_matrix:
        """
        Extract gene expression for cells at source timepoint(s).
        
        Parameters
        ----------
        adata : AnnData
            Annotated data matrix with cell metadata
        days_t0 : List[int], optional
            List of source days/timepoints. If None, uses all unique days in sorted order.
        day_column_name : str, default "time_info"
            Column name in adata.obs containing day/timepoint information
        
        Returns
        -------
        x_0 : csr_matrix
            Gene expression matrix for source cells, shape (n_source_cells, n_genes)
        """
        if days_t0 is None:
            days_t0 = sorted(adata.obs[day_column_name].unique())[:-1]
        
        mask = adata.obs[day_column_name].isin(days_t0)
        x_0 = adata[mask].X
        
        return x_0

    def flow_map_at_day_t0(
        self,
        adata: ad.AnnData,
        tmap: csr_matrix,
        days_t0: Optional[List[int]] = None,
        day_column_name: str = "time_info",
    ) -> csr_matrix:
        """
        Construct flow map by transporting source cell expressions through the transport map.
        
        The flow map represents predicted gene expressions of cells at days_t0 after
        transport to days_t1, computed as:
        
            x_t = T[source_indices, :] @ adata.X
        
        where T is the global transport map and source_indices selects rows corresponding
        to cells in days_t0.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data matrix with cell metadata
        tmap : csr_matrix
            Global transport map of shape (n_cells, n_cells) with block structure:
            - Non-zero entries only for sequential day transitions
            - Each row represents a source cell
            - Each column represents a target cell
        days_t0 : List[int], optional
            Source days/timepoints. If None, uses all unique days in sorted order.
        days_t1 : List[int], optional
            Target days/timepoints. Used for validation/filtering. If None, inferred from days_t0.
        day_column_name : str, default "time_info"
            Column name in adata.obs containing day/timepoint information
        
        Returns
        -------
        x_t : csr_matrix
            Transported gene expression matrix of shape (n_source_cells, n_genes),
            representing predicted gene expressions of cells at days_t0 after
            transport to days_t1
        """
        if days_t0 is None:
            days_t0 = sorted(adata.obs[day_column_name].unique())[:-1]
        

        source_mask = adata.obs[day_column_name].isin(days_t0)
        source_indices = np.where(source_mask)[0]
        
        tmap_subset = tmap[source_indices, :]        
        x_t = tmap_subset @ adata.X
        
        return x_t
    

def get_day_t0_expression(
    adata: ad.AnnData,
    days_t0: Optional[List[int]] = None,
    day_column_name: str = "time_info",
) -> csr_matrix:
    """
    Functional wrapper to extract gene expression for cells at source timepoint(s).
    """
    return FlowMap().day_t0_expression(
        adata=adata,
        days_t0=days_t0,
        day_column_name=day_column_name,
    )


def get_flow_map(
    adata: ad.AnnData,
    tmap: csr_matrix,
    days_t0: Optional[List[int]] = None,
    day_column_name: str = "time_info",
) -> csr_matrix:
    """
    Functional wrapper to construct flow map by transporting source cell expressions.
    """
    return FlowMap().flow_map_at_day_t0(
        adata=adata,
        tmap=tmap,
        days_t0=days_t0,
        day_column_name=day_column_name,
    )