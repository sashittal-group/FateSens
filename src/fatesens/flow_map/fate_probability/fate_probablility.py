from typing import Optional, List
import os
import wot


def calculate_fate_probability(
    adata,
    tmap_path="tmaps",
    day_column_name="time_info",
    state_column_name="state_info",
    day_t0: Optional[int] = None,
    day_t1: Optional[int] = None,
    final_fates=["Monocyte", "Neutrophil"],
    target_states=["Monocyte", "Neutrophil"],
):
    """
    Calculate fate probabilities for cells using a transport map model.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object with time and state information in obs.
    tmap_path : str
        Path to the WOT transport map model directory.
    day_column_name : str, optional
        Column name in adata.obs for time information. Default is "time_info".
    state_column_name : str, optional
        Column name in adata.obs for state information. Default is "state_info".
    day_t0 : Optional[int], optional
        Start timepoint. If None, uses minimum value.
    day_t1 : Optional[int], optional
        End timepoint for target cells. If None, uses maximum value.
    final_fates : list, optional
        List of final fate states to normalize probabilities for. If None, uses all states.
    target_states : list, optional
        Specific states to extract probabilities for. If None, uses all states.
    
    Returns
    -------
    pd.DataFrame
        Normalized fate probabilities for each state.
    """
    tmap_model_path_wot = os.path.join(tmap_path, "serum")
    if not os.path.exists(tmap_path):
        os.makedirs(tmap_path, exist_ok=True)
        ot_model = wot.ot.OTModel(
            adata, 
            day_field=day_column_name, 
            epsilon=0.05, 
            lambda1=1, 
            lambda2=50, 
            growth_iters=3
        )
        ot_model.compute_all_transport_maps(tmap_out=tmap_model_path_wot)
    
    # Load the transport map model
    tmap_model = wot.tmap.TransportMapModel.from_directory(tmap_model_path_wot)
    
    # Set default time values if not provided
    if day_t0 is None:
        day_t0 = [adata.obs[day_column_name].min()]
    if day_t1 is None:
        day_t1 = [adata.obs[day_column_name].max()]

    # Get target cells at the end timepoint
    target_cells = adata[adata.obs[day_column_name].isin(day_t1)]
    
    cell_sets = {
        state: target_cells.obs_names[target_cells.obs[state_column_name] == state]
        for state in target_cells.obs[state_column_name].unique()
    }
    
    # Calculate populations from cell sets
    populations = tmap_model.population_from_cell_sets(cell_sets, at_time=day_t1[-1])
    
    # Compute trajectories and fate probabilities
    trajectory_ds = tmap_model.trajectories(populations)
    fate_probabilities = trajectory_ds.to_df()
    
    # Select target states if specified
    if target_states is not None:
        fate_probabilities = fate_probabilities[target_states]
    
    # Normalize fate probabilities
    row_sums = fate_probabilities.sum(axis=1) + 1e-12
    fate_normalized = fate_probabilities.div(row_sums, axis=0)
    
    # Handle rows with zero sum by assigning equal probabilities
    if final_fates is not None:
        zero_mask = fate_normalized.sum(axis=1) == 0
        fate_normalized.loc[zero_mask, final_fates] = 1.0 / len(final_fates)
    
    fate_normalized = fate_normalized.loc[adata.obs_names.values]
    fate_normalized = fate_normalized[adata.obs[day_column_name].isin(day_t0)]
    return fate_normalized.values