import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
from typing import List, Tuple, Optional
from scipy import stats
from scipy.sparse import csr_matrix
from collections import defaultdict
from statsmodels.stats.multitest import multipletests
import wot
import os


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

def calculate_mean_sensitivity(adata: ad.AnnData, sensitivities: np.array, days_t0: List[int], cell_type_key = "state_info", cell_types = ["Undifferentiated"]):
    adata = adata[adata.obs["time_info"].isin(days_t0)]
    indices = adata.obs[cell_type_key].isin(cell_types)
    mean_sensitivity = np.mean(sensitivities[indices], axis=0)
    return mean_sensitivity

def calculate_sensitivity_scores_stats(adata: ad.AnnData, sensitivities: np.array, days_t0: List[int], cell_type_key: str = "state_info", cell_types: List[str] = ["Undifferentiated"], multipletests_method: str = 'bonferroni') -> pd.DataFrame:
    adata_filtered = adata[adata.obs["time_info"].isin(days_t0)].copy()
    indices = adata_filtered.obs[cell_type_key].isin(cell_types)
    sensitivities = np.atleast_2d(np.squeeze(sensitivities))
    sensitivities = sensitivities[indices]
    
    mean_scores = np.mean(sensitivities, axis=0)
    
    _, p_values = stats.ttest_1samp(sensitivities, popmean=0, alternative='two-sided')
    p_values = np.nan_to_num(p_values, nan=1.0)
    
    _, p_adj, _, _ = multipletests(p_values, method=multipletests_method)
    results_df = pd.DataFrame({
        'gene': adata.var_names,
        'mean_sensitivity_score': mean_scores,
        'p-value': p_values,
        'fdr_adj_p_value': p_adj
    })
    
    return results_df


def get_2_type_of_clonal_trajectory(
    adata,
    cell_type_column="state_info",
    clone_column="X_clone",
    embedding_key="X_emb",
    root_cell_type="Undifferentiated",
    fate1_cell_type="Neutrophil",
    fate2_cell_type="Monocyte",
    N=3,
    n_neighbors=30,
    median_offset=0.09,
):
    """Compute clonal trajectories and classify cells based on fate composition.
    
    Computes both raw and smoothed fate scores for root cells based on their sister cells,
    then performs fate classification for both score types.
    """
    X_clone = adata.obsm[clone_column].tocsc()
    n_cells = X_clone.shape[0]

    clone_to_cells = defaultdict(list)
    for clone_id in range(X_clone.shape[1]):
        cells = X_clone[:, clone_id].nonzero()[0]
        if len(cells) > 1:
            clone_to_cells[clone_id] = cells.tolist()

    cell_to_sisters = defaultdict(set)
    for cells in clone_to_cells.values():
        for i in cells:
            cell_to_sisters[i].update(cells)
    for i in cell_to_sisters:
        cell_to_sisters[i].discard(i)

    cell_types = adata.obs[cell_type_column].values

    is_root_cell_type = cell_types == root_cell_type
    is_fate1_cell_type = cell_types == fate1_cell_type
    is_fate2_cell_type = cell_types == fate2_cell_type

    raw_score = np.zeros(n_cells)

    for i in np.where(is_root_cell_type)[0]:
        sisters = list(cell_to_sisters.get(i, []))
        sisters = [j for j in sisters if (not is_root_cell_type[j] and (is_fate1_cell_type[j] or is_fate2_cell_type[j]))]
        if len(sisters) == 0:
            raw_score[i] = 0.0
            continue
        n_neu = sum(is_fate1_cell_type[j] for j in sisters)
        n_dc = sum(is_fate2_cell_type[j] for j in sisters)
        denom = len(sisters)
        if (n_neu + n_dc) == 0:
            raw_score[i] = 0.0
        else:
            raw_score[i] = (n_neu / denom) - (n_dc / denom)

    adata.obs[f"{fate1_cell_type}_{fate2_cell_type}_raw_score"] = raw_score

    # build kNN graph if not already present
    if "connectivities" not in adata.obsp:
        sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=n_neighbors)

    W = adata.obsp["connectivities"].tocsr()
    row_sum = np.array(W.sum(axis=1)).flatten()
    row_sum[row_sum == 0] = 1
    W = csr_matrix(W.multiply(1 / row_sum[:, None]))

    smoothed = raw_score.copy()
    for _ in range(N):
        smoothed = W @ smoothed

    adata.obs[f"{fate1_cell_type}_{fate2_cell_type}_smoothed_score"] = smoothed

    # Clustering for both raw and smoothed scores
    for score_type, score in zip(["raw", "smoothed"], [raw_score, smoothed]):
        mono_scores = score[is_root_cell_type]
        median_score = np.median(mono_scores)
        print(f"{score_type} median: {median_score}")
        mono_class = np.full(n_cells, "NA", dtype=object)
        mono_class[(is_root_cell_type) & (score > median_score + median_offset)] = f"{fate1_cell_type}-like"
        mono_class[(is_root_cell_type) & (score <= median_score - median_offset)] = f"{fate2_cell_type}-like"
        adata.obs[f"{fate1_cell_type}_{fate2_cell_type}_{score_type}_fate_class"] = mono_class

    return adata


def compute_fate_trajectories_and_diff_exp_wot(adata, tmap_model_path="tmaps", 
                                           day_column_name='time_info', 
                                           state_field='state_info',
                                           fates=["Monocyte", "Neutrophil"],
                                           days_t0: List[int] = [2, 4],):
    """
    Compute fate trajectories from a transport map model and perform differential expression analysis.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with temporal and state information
    wot_tmap_path : str
        Path to the directory containing the transport map model
    day_column_name : str, default='time_info'
        Column in adata.obs containing time information
    state_field : str, default='state_info'
        Column in adata.obs containing cell state information
    fates : list or None
        List of fate types to compute probabilities for. If None, uses ['Monocyte', 'Neutrophil']
    days_t0 : list or None
        Time points to use for differential expression analysis. If None, uses [2, 4]
    
    Returns
    -------
    pd.DataFrame
         DataFrame containing differential expression results for the specified fates and time points
    """
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

    # Get the last day in the dataset
    last_day = adata.obs[day_column_name].max()
    
    # Extract target cells at the last timepoint
    target_cells = adata[adata.obs[day_column_name] == last_day]
    
    # Create cell sets from states at final timepoint
    cell_sets = {
        state: target_cells.obs_names[target_cells.obs[state_field] == state]
        for state in target_cells.obs[state_field].unique()
    }
    
    # Compute populations from cell sets
    populations = tmap_model.population_from_cell_sets(cell_sets, at_time=last_day)
    
    # Compute trajectories and get fate probabilities
    trajectory_ds = tmap_model.trajectories(populations)
    fate_probabilities = trajectory_ds.to_df()
    
    # Filter to specified fates if they exist in the data
    fates_available = [f for f in fates if f in fate_probabilities.columns]
    if not fates_available:
        raise ValueError(f"None of the specified fates {fates} found in trajectory data. "
                        f"Available fates: {list(fate_probabilities.columns)}")
    
    fate_probabilities_filtered = fate_probabilities[fates_available]
    
    # Normalize fate probabilities
    row_sums = fate_probabilities_filtered.sum(axis=1) + 1e-12
    fate_normalized = fate_probabilities_filtered.div(row_sums, axis=0)
    
    # Compute fates for all cells
    all_cell_sets = {
        state: adata.obs_names[adata.obs[state_field] == state]
        for state in adata.obs[state_field].unique()
    }
    target_cell_set = tmap_model.population_from_cell_sets(all_cell_sets, at_time=last_day)
    fate_ds = tmap_model.fates(target_cell_set)
    
    # Perform differential expression analysis on specified time points
    adata_subset = adata[adata.obs[day_column_name].isin(days_t0)]
    diff_exp_results = wot.tmap.diff_exp(
        adata_subset, 
        fate_ds,
        cell_days_field=day_column_name, 
        compare='all'
    )
    
    return diff_exp_results

def filter_and_select_tf_wot(results: pd.DataFrame, root_celltype: str="Undifferentiated", final_celltype: str="Monocyte", all_days: List[int]=[2, 4, 6]) -> pd.DataFrame:
    filtered = results[
        (results["t_fdr"] < 0.05)
        & (results["name1"] == root_celltype)
        & (results["name2"] == final_celltype)
        & (results["day1"].isin(all_days))
    ]
    unique_results = (
        filtered.sort_values("fraction_expressed_ratio", ascending=True)
        .groupby(level=0)
        .first()
        .sort_values("fraction_expressed_ratio", ascending=True)
    )
    return unique_results.index.tolist()

def get_ground_truth_regulatory_degs(
    adata: ad.AnnData,
    state_key: str = "state_info",
    time_key: str = "time_info",
    fate_class_key: str = "Neutrophil_Monocyte_fate_class",
    terminal_fate: str = "Monocyte-like",
    root_state: str = "Undifferentiated",
    days_t0: List[int] = [2, 4],
    pval_cutoff: float = 0.05,
    log2fc_min: float = 0.5,
) -> List[str]:
    adata_sub = adata[
        adata.obs[state_key].isin([root_state])
        & adata.obs[time_key].isin(days_t0)
        & ~adata.obs[fate_class_key].isin(["NA"])
    ].copy()

    sc.tl.rank_genes_groups(
        adata_sub,
        fate_class_key,
        method="wilcoxon",
        key_added="rank_genes_groups_results",
    )
    result_df = sc.get.rank_genes_groups_df(
        adata_sub,
        group=None,
        key="rank_genes_groups_results",
        pval_cutoff=pval_cutoff,
        log2fc_min=log2fc_min,
    )
    result_df = result_df[result_df["group"] == terminal_fate]
    return result_df.sort_values(by="scores", ascending=False)["names"].tolist()
