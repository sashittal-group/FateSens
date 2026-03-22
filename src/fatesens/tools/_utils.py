import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
from typing import List, Tuple, Optional


def get_marker_gene_for_fate(adata, fate, all_fates=["Monocyte", "Neutrophil"], 
                              n_top_markers=None, pval_cutoff=0.05, 
                              log2fc_min=0.5, state_key='state_info'):
    """
    Identify marker genes for a specific cell fate/state using differential gene expression analysis.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with gene expression data
    fate : str
        The specific fate/state to identify marker genes for
    all_fates : list
        List of all cell fates/states to use for comparison
    n_top_markers : int or None
        Number of top markers to return. If None, returns all markers
    pval_cutoff : float
        Adjusted p-value cutoff for filtering results
    log2fc_min : float
        Minimum log2 fold change for filtering results
    state_key : str
        Key in adata.obs containing state/cluster assignments
        
    Returns
    -------
    list
        List of top marker genes for the specified fate
    """
    
    # Filter for all specified fates
    adata_filtered = adata[adata.obs[state_key].isin(all_fates)].copy()
    
    # Perform differential expression analysis
    sc.tl.rank_genes_groups(
        adata_filtered,
        state_key,
        method='wilcoxon',
        key_added='rank_genes_groups_results'
    )
    
    # Get results as DataFrame
    result_df = sc.get.rank_genes_groups_df(
        adata_filtered,
        group=None,
        key='rank_genes_groups_results',
        pval_cutoff=pval_cutoff,
        log2fc_min=log2fc_min
    )
    
    # Extract top markers for the specified fate
    fate_markers = result_df[result_df['group'] == fate].sort_values(
        by='scores', ascending=False
    )
    if n_top_markers is None:
        top_markers = fate_markers['names'].tolist()
    else:
        top_markers = fate_markers.head(n_top_markers)['names'].tolist()
    
    return top_markers

def get_filter_matrix(adata: ad.AnnData, marker_genes):
    marker_indices = adata.var_names.get_indexer(marker_genes)
    F = np.zeros((len(marker_indices), adata.n_vars))
    for i, idx in enumerate(marker_indices):
        F[i, idx] = 1
    return F

def calculate_mean_sensitivity(adata: ad.AnnData, sensitivities: np.array, days_t0: Optional[List[int]] = None, cell_type_key = "state_info", cell_types = ["Undifferentiated"]):
    adata = adata[adata.obs["time_info"].isin(days_t0)]
    indices = adata.obs[cell_type_key].isin(cell_types)
    mean_sensitivity = np.mean(sensitivities[indices], axis=0)
    return mean_sensitivity