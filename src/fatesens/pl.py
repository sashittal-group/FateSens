import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import adjustText as ta

try:
    import textalloc as textalloc_lib
except ImportError:
    textalloc_lib = None


def plot_sensitivity_volcano(df, p_thresh=0.05, score_thresh=0.005, top_n=30):
    """
    Plot volcano plot for sensitivity scores.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results DataFrame from calculate_sensitivity_scores_stats with columns:
        'gene', 'mean_sensitivity_score', 'p-value', 'fdr_adj_p_value'
    p_thresh : float
        P-value threshold for significance
    score_thresh : float
        Sensitivity score threshold for significance
    top_n : int
        Number of top genes to label (split between up and down)
    """
    plot_df = df.copy()
    
    plot_df['-log10_p'] = -np.log10(plot_df['fdr_adj_p_value'].replace(0, 1e-300))
    
    plot_df['status'] = 'NS'
    plot_df.loc[(plot_df['fdr_adj_p_value'] < p_thresh) & (plot_df['mean_sensitivity_score'] > score_thresh), 'status'] = 'UP'
    plot_df.loc[(plot_df['fdr_adj_p_value'] < p_thresh) & (plot_df['mean_sensitivity_score'] < -score_thresh), 'status'] = 'DOWN'
    
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['axes.linewidth'] = 1.2
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.set_style("ticks")
    
    ns_data = plot_df[plot_df['status'] == 'NS']
    up_data = plot_df[plot_df['status'] == 'UP']
    down_data = plot_df[plot_df['status'] == 'DOWN']
    
    ax.scatter(
        ns_data['mean_sensitivity_score'], ns_data['-log10_p'],
        color='#CCCCCC', alpha=0.3, s=20, zorder=1, label='Not Significant', edgecolors='none'
    )
    
    ax.scatter(
        down_data['mean_sensitivity_score'], down_data['-log10_p'],
        color='#4DBBD5', alpha=0.85, s=45, zorder=2, label='Neutrophil Significant\nRegulatory Gene', 
        edgecolors='white', linewidths=0.5
    )
    
    ax.scatter(
        up_data['mean_sensitivity_score'], up_data['-log10_p'],
        color='#E64B35', alpha=0.85, s=45, zorder=2, label='Monocyte Significant\nRegulatory Gene', 
        edgecolors='white', linewidths=0.5
    )
    
    ax.axhline(-np.log10(p_thresh), color='#555555', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    ax.axvline(score_thresh, color='#555555', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    ax.axvline(-score_thresh, color='#555555', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
    
    top_up = up_data.sort_values('mean_sensitivity_score', ascending=False).head(top_n // 2)
    top_down = down_data.sort_values('mean_sensitivity_score', ascending=True).head(top_n // 2)
    label_df = pd.concat([top_up, top_down])
    
    if not label_df.empty:
        texts = []
        for idx, row in label_df.iterrows():
            text = ax.text(
                row['mean_sensitivity_score'],
                row['-log10_p'],
                row['gene'],
                fontsize=10,
                ha='center'
            )
            texts.append(text)
        
        ta.adjust_text(
            texts,
            x=plot_df['mean_sensitivity_score'].values,
            y=plot_df['-log10_p'].values,
            arrowprops=dict(arrowstyle='-', color='#777777', lw=0.8),
            expand_points=(1.2, 1.2)
        )
    
    ax.set_xlabel('Mean Sensitivity Score', fontsize=14, labelpad=10)
    ax.set_ylabel('-log10 (FDR Adjusted P-value)', fontsize=14, labelpad=10)
    
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.2, length=5)
    
    legend = ax.legend(frameon=True, fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    legend.get_frame().set_linewidth(1.2)
    
    sns.despine(ax=ax, top=True, right=True)
    
    plt.tight_layout()
    plt.show()



def get_quadrant_top_genes(
    adata,
    mean_sensitivities_fate1,
    mean_sensitivities_fate2,
    fate1_name: str = "Fate1",
    fate2_name: str = "Fate2",
    top_n: int = 10,
    penalty_weight: float = 1.5,
):
    """
    Compute top genes for each quadrant in the sensitivity comparison plot.
    Returns: q1_top, q2_top, q3_top, q4_top (DataFrames)
    """
    df = pd.DataFrame({
        'gene': adata.var_names,
        f'{fate1_name}_score': np.atleast_1d(np.squeeze(mean_sensitivities_fate1)),
        f'{fate2_name}_score': np.atleast_1d(np.squeeze(mean_sensitivities_fate2))
    })
    df['distance'] = np.sqrt(df[f'{fate1_name}_score']**2 + df[f'{fate2_name}_score']**2)

    # Q1: +, +
    q1_df = df[(df[f'{fate1_name}_score'] > 0) & (df[f'{fate2_name}_score'] > 0)].copy()
    q1_df['custom_score'] = q1_df['distance'] - (abs(q1_df[f'{fate1_name}_score'] - q1_df[f'{fate2_name}_score']) * penalty_weight)
    q1_top = q1_df.nlargest(top_n, 'custom_score')

    # Q2: -, +
    q2_df = df[(df[f'{fate1_name}_score'] < 0) & (df[f'{fate2_name}_score'] > 0)].copy()
    q2_df['penalty'] = np.minimum(abs(q2_df[f'{fate1_name}_score']), abs(q2_df[f'{fate1_name}_score'] + q2_df[f'{fate2_name}_score']))
    q2_df['custom_score'] = q2_df['distance'] - (q2_df['penalty'] * penalty_weight)
    q2_top = q2_df.nlargest(top_n, 'custom_score')

    # Q3: -, -
    q3_df = df[(df[f'{fate1_name}_score'] < 0) & (df[f'{fate2_name}_score'] < 0)].copy()
    q3_df['custom_score'] = q3_df['distance'] - (abs(q3_df[f'{fate1_name}_score'] - q3_df[f'{fate2_name}_score']) * penalty_weight)
    q3_top = q3_df.nlargest(top_n, 'custom_score')

    # Q4: +, -
    q4_df = df[(df[f'{fate1_name}_score'] > 0) & (df[f'{fate2_name}_score'] < 0)].copy()
    q4_df['penalty'] = np.minimum(abs(q4_df[f'{fate2_name}_score']), abs(q4_df[f'{fate1_name}_score'] + q4_df[f'{fate2_name}_score']))
    q4_df['custom_score'] = q4_df['distance'] - (q4_df['penalty'] * penalty_weight)
    q4_top = q4_df.nlargest(top_n, 'custom_score')

    return q1_top, q2_top, q3_top, q4_top


def plot_sensitivity_comparison(
    adata,
    mean_sensitivities_fate1,
    mean_sensitivities_fate2,
    fate1_name: str = "Fate1",
    fate2_name: str = "Fate2",
    top_n: int = 10,
    penalty_weight: float = 1.5,
):
    """
    Plot comparison of mean sensitivity scores for two fates in a 2D scatter plot.
    """
    q1_top, q2_top, q3_top, q4_top = get_quadrant_top_genes(
        adata,
        mean_sensitivities_fate1,
        mean_sensitivities_fate2,
        fate1_name=fate1_name,
        fate2_name=fate2_name,
        top_n=top_n,
        penalty_weight=penalty_weight,
    )
    df = pd.DataFrame({
        'gene': adata.var_names,
        f'{fate1_name}_score': np.atleast_1d(np.squeeze(mean_sensitivities_fate1)),
        f'{fate2_name}_score': np.atleast_1d(np.squeeze(mean_sensitivities_fate2))
    })
    top_genes = pd.concat([q1_top, q2_top, q3_top, q4_top])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(df[f'{fate1_name}_score'], df[f'{fate2_name}_score'],
               color='#DFDFDF', alpha=0.5, s=20, edgecolors='none',
               label='Background genes', zorder=1)
    ax.scatter(top_genes[f'{fate1_name}_score'], top_genes[f'{fate2_name}_score'],
               color='#D55E00', s=70, alpha=0.95, edgecolors='white', linewidth=1.0,
               label=f'Top {len(top_genes)} driver genes', zorder=3)
    ax.axhline(0, color='#333333', linestyle='-', linewidth=1, zorder=2)
    ax.axvline(0, color='#333333', linestyle='-', linewidth=1, zorder=2)
    ax.axline((0, 0), slope=1, color='#888888', linestyle='--', alpha=0.5, linewidth=1.2, zorder=2)
    ax.axline((0, 0), slope=-1, color='#888888', linestyle=':', alpha=0.6, linewidth=1.2, zorder=2)
    if textalloc_lib is not None and not top_genes.empty:
        x_text = top_genes[f'{fate1_name}_score'].values
        y_text = top_genes[f'{fate2_name}_score'].values
        text_list = top_genes['gene'].tolist()
        textalloc_lib.allocate_text(
            fig, ax,
            x=x_text,
            y=y_text,
            text_list=text_list,
            x_scatter=df[f'{fate1_name}_score'].values,
            y_scatter=df[f'{fate2_name}_score'].values,
            textsize=12,
            draw_lines=True,
            linewidth=0.6,
            linecolor='#666666',
            margin=0.015
        )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
    ax.set_xlabel(f'{fate1_name} Lineage Mean Sensitivity Score', fontsize=14, labelpad=12)
    ax.set_ylabel(f'{fate2_name} Lineage Mean Sensitivity Score', fontsize=14, labelpad=12)
    ax.grid(True, linestyle='-', alpha=0.15, zorder=0)
    ax.legend(frameon=False, markerscale=1.5, loc='upper left', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_enrichment_matrix_dotplot(
    adata,
    mean_sensitivities_fate1,
    mean_sensitivities_fate2,
    fate1_name: str = "Fate1",
    fate2_name: str = "Fate2",
    top_n: int = 30,
    penalty_weight: float = 1.5,
    gene_sets=['MSigDB_Hallmark_2020'],
    organism='mouse',
    out_svg=None
):
    """
    Plot an advanced enrichment matrix dot plot for top genes in each quadrant.
    Quadrant genes are computed internally using get_quadrant_top_genes.
    """
    import pandas as pd
    import numpy as np
    import gseapy as gp
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec

    q1_top, q2_top, q3_top, q4_top = get_quadrant_top_genes(
        adata,
        mean_sensitivities_fate1,
        mean_sensitivities_fate2,
        fate1_name=fate1_name,
        fate2_name=fate2_name,
        top_n=top_n,
        penalty_weight=penalty_weight,
    )


    # Use the 'gene' column for gene names (as strings) for gseapy
    gene_dict = {
        '+, +': [str(g) for g in q1_top['gene'].tolist()],
        '-, +': [str(g) for g in q2_top['gene'].tolist()],
        '-, -': [str(g) for g in q3_top['gene'].tolist()],
        '+, -': [str(g) for g in q4_top['gene'].tolist()]
    }

    all_results = []
    for group_name, gene_list in gene_dict.items():
        if not gene_list:
            continue
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            organism=organism,
            # background=adata.var_names if adata is not None else None,
            outdir=None
        )
        res = enr.res2d.copy()
        if not res.empty:
            res['Group'] = group_name
            all_results.append(res)

    if not all_results:
        raise ValueError("No enrichment results found. Check your gene names.")

    res_df = pd.concat(all_results, ignore_index=True)

    # Dynamically find columns
    overlap_col = next((col for col in res_df.columns if 'overlap' in col.lower()), None)
    term_col = next((col for col in res_df.columns if 'term' in col.lower()), None)
    adj_pval_col = next((col for col in res_df.columns if 'adj' in col.lower() and 'p-value' in col.lower()), None)
    pval_col = next((col for col in res_df.columns if 'p-value' in col.lower() and 'adj' not in col.lower()), None)

    def calc_ratio(overlap_str):
        if pd.isna(overlap_str): return 0
        if isinstance(overlap_str, (int, float)): return float(overlap_str)
        num, den = str(overlap_str).split('/')
        return int(num) / int(den)

    res_df['Overlap_Ratio'] = res_df[overlap_col].apply(calc_ratio)
    res_df[term_col] = res_df[term_col].apply(lambda x: x[:45] + '...' if len(x) > 45 else x)
    top_terms = (res_df.sort_values(['Group', pval_col])
                       .groupby('Group')
                       .head(5))

    groups_order = ['+, +', '-, +', '-, -', '+, -']
    x_mapping = {g: i for i, g in enumerate(groups_order)}
    unique_pathways = top_terms[term_col].unique().tolist()
    y_mapping = {pathway: i for i, pathway in enumerate(reversed(unique_pathways))}
    top_terms['x'] = top_terms['Group'].map(x_mapping)
    top_terms['y'] = top_terms[term_col].map(y_mapping)

    size_multiplier = 10000
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 1, height_ratios=[5, 1], hspace=0.02)
    ax_main = fig.add_subplot(gs[0])
    ax_matrix = fig.add_subplot(gs[1], sharex=ax_main)
    custom_colors = ["#FF0000", "#AA00AA", "#0000FF"]
    cmap = mcolors.LinearSegmentedColormap.from_list("red_purple_blue", custom_colors)
    scatter = ax_main.scatter(
        top_terms['x'],
        top_terms['y'],
        s=top_terms['Overlap_Ratio'] * size_multiplier,
        c=top_terms[adj_pval_col],
        cmap=cmap,
        alpha=0.9,
        edgecolors='none'
    )
    ax_main.set_yticks(range(len(unique_pathways)))
    ax_main.set_yticklabels(reversed(unique_pathways), fontsize=14)
    ax_main.set_xlim(-0.5, len(groups_order) - 0.5)
    ax_main.set_ylim(-0.5, len(unique_pathways) - 0.5)
    ax_main.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_main.spines['top'].set_visible(True)
    ax_main.spines['right'].set_visible(True)
    ax_main.spines['bottom'].set_visible(True)
    ax_main.set_axisbelow(True)
    ax_main.grid(True, axis='both', linestyle='--', alpha=0.4)
    matrix_labels = ["Monocyte", "Neutrophil"]
    y_matrix_coords = [1, 0]
    for col_idx, group in enumerate(groups_order):
        signs = [s.strip() for s in group.split(',')]
        mono_high = (signs[0] == '+')
        neutro_high = (signs[1] == '+')
        conditions = [mono_high, neutro_high]
        for row_idx, is_high in enumerate(conditions):
            y_val = y_matrix_coords[row_idx]
            if is_high:
                ax_matrix.scatter(col_idx, y_val, s=150, c='black', edgecolors='black', zorder=3)
            else:
                ax_matrix.scatter(col_idx, y_val, s=150, c='white', edgecolors='black', linewidths=1.5, zorder=3)
    ax_matrix.set_yticks(y_matrix_coords)
    ax_matrix.set_yticklabels(matrix_labels, fontsize=14)
    ax_matrix.set_ylim(-0.8, 1.8)
    for spine in ax_matrix.spines.values():
        spine.set_visible(False)
    ax_matrix.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False)
    for col_idx in range(len(groups_order)):
        ax_matrix.hlines(y=-0.5, xmin=col_idx - 0.2, xmax=col_idx + 0.2, color='black', linewidth=1.5)
    plt.subplots_adjust(right=0.82)
    leg_x = 0.86
    leg_ax = fig.add_axes([leg_x, 0.60, 0.1, 0.25])
    leg_ax.axis('off')
    leg_ax.text(0.0, 1.0, "Overlap Ratio", ha='left', va='top', fontsize=14, transform=leg_ax.transAxes)
    min_ratio = round(top_terms['Overlap_Ratio'].min(), 2)
    max_ratio = round(top_terms['Overlap_Ratio'].max(), 2)
    mid_ratio = round((min_ratio + max_ratio) / 2, 2)
    ratios_to_show = sorted(list(set([min_ratio, mid_ratio, max_ratio])))
    for i, val in enumerate(ratios_to_show):
        y_pos = 0.75 - (i * 0.25)
        leg_ax.scatter(0.2, y_pos, s=val * size_multiplier, c='gray', alpha=0.6, transform=leg_ax.transAxes)
        leg_ax.text(0.45, y_pos, f"{val:.2f}", va='center', ha='left', fontsize=14, transform=leg_ax.transAxes)
    cax = fig.add_axes([leg_x + 0.01, 0.40, 0.02, 0.15])
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.ax.invert_yaxis()
    cax.text(0.0, 1.1, 'Adj P-value', transform=cax.transAxes, ha='left', va='bottom', fontsize=14)
    if out_svg:
        plt.savefig(out_svg, bbox_inches='tight')
    plt.show()

def plot_iou_concordance(gene_lists, method_names, reference_method_idx=0, figsize=(8, 5)):
    """
    Plot IoU (Intersection over Union) concordance between two methods.
    
    Compares gene rankings from two methods by computing IoU at different k values
    (number of top genes considered).
    
    Parameters
    ----------
    gene_lists : list of list of str
        Three lists of genes in sorted order (by ranking score):
        - gene_lists[0]: first method genes
        - gene_lists[1]: second method genes  
        - gene_lists[2]: reference genes (used for comparison)
    method_names : list of str
        Names of the two methods to compare, e.g., ['FateSens', 'Waddington-OT']
    reference_method_idx : int, default=0
        Index indicating which method is the reference (0, 1, or 2). 
        The reference is used for comparison against the other methods.
    figsize : tuple, default=(8, 5)
        Figure size (width, height)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    
    if len(gene_lists) != 3:
        raise ValueError(f"Expected 3 gene lists, got {len(gene_lists)}")
    if len(method_names) != 2:
        raise ValueError(f"Expected 2 method names, got {len(method_names)}")
    
    # Assign reference genes
    ref_genes = gene_lists[reference_method_idx]
    
    # Assign comparison methods (exclude reference)
    remaining_indices = [i for i in range(3) if i != reference_method_idx]
    method1_genes = gene_lists[remaining_indices[0]]
    method2_genes = gene_lists[remaining_indices[1]]
    
    # Compute IoU across k values
    max_k = min(len(ref_genes), len(method1_genes), len(method2_genes))
    k_vals = list(range(1, max_k + 1))
    
    iou_method1 = []
    iou_method2 = []
    
    for k in k_vals:
        ref_top_k = set(ref_genes[:k])
        method1_top_k = set(method1_genes[:k])
        method2_top_k = set(method2_genes[:k])
        
        # Compute IoU: |intersection| / |union|
        iou_m1 = len(ref_top_k & method1_top_k) / len(ref_top_k | method1_top_k)
        iou_m2 = len(ref_top_k & method2_top_k) / len(ref_top_k | method2_top_k)
        
        iou_method1.append(iou_m1)
        iou_method2.append(iou_m2)
    
    # Calculate normalized AUC
    auc_method1 = sum(iou_method1) / len(k_vals)
    auc_method2 = sum(iou_method2) / len(k_vals)
    
    # Create figure with single panel
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colorblind-safe palette
    color_method1 = '#D55E00'  # Vermillion
    color_method2 = '#0072B2'  # Deep Blue
    
    # Plot both methods
    ax.plot(k_vals, iou_method1, color=color_method1, linestyle='-', linewidth=2.5, 
            alpha=0.9, label=method_names[0])
    ax.plot(k_vals, iou_method2, color=color_method2, linestyle='-', linewidth=2.5, 
            alpha=0.9, label=method_names[1])
    
    # Add normalized AUC scores as text box
    auc_text = f"{method_names[0]} AUC: {auc_method1:.3f} | {method_names[1]} AUC: {auc_method2:.3f}"
    ax.text(0.95, 0.05, auc_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#cccccc'))
    
    # Labels and legend
    ax.set_xlabel('Number of Top Genes Considered (k)', fontsize=12)
    ax.set_ylabel('IoU with Reference Gene List', fontsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False, fontsize=10)
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    ax.grid(True, linestyle='-', alpha=0.15, zorder=0)
    
    plt.tight_layout()
    plt.show()


def plot_ridge_on_ftle(
    adata,
    ridge_indices,
    ftle,
    day_t0,
    state_key="state_info",
    state_value="Undifferentiated",
    time_key="time_info",
    background_cmap="Reds",
    ridge_color="Green",
    background_s=5,
    ridge_s=5,
):
    """
    Plot ridge cells on top of FTLE (Finite-Time Lyapunov Exponent) background.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with obsm["X_emb"] containing coordinates.
    ridge_indices : list of array-like
        Ridge cell indices from estimate_ridge (list of cell index arrays).
    ftle : array-like of shape (n_cells,)
        FTLE values for coloring the background.
    day_t0 : tuple or list
        Time points to filter (e.g., [2, 4]).
    state_key : str, default="state_info"
        Column name in adata.obs for cell state.
    state_value : str, default="Undifferentiated"
        State value to highlight on ridge.
    time_key : str, default="time_info"
        Column name in adata.obs for time/day information.
    background_cmap : str, default="Reds"
        Colormap for FTLE background.
    ridge_color : str, default="Green"
        Color for ridge cells.
    background_s : int, default=5
        Marker size for background cells.
    ridge_s : int, default=5
        Marker size for ridge cells.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    time_values = day_t0

    # Filter cells by time
    adata_filtered = adata[adata.obs[time_key].isin(time_values)]

    # Get ridge cells
    ridge_adata_all_obs = ridge_indices[0].tolist()
    adata_on_ridge = adata_filtered[adata_filtered.obs.index.isin(ridge_adata_all_obs)].copy()

    # Create figure
    fig, ax = plt.subplots()

    # Plot FTLE background
    scatter_bg = ax.scatter(
        adata_filtered.obsm["X_emb"][:, 0],
        adata_filtered.obsm["X_emb"][:, 1],
        c=ftle,
        cmap=background_cmap,
        s=background_s,
        alpha=0.7,
        zorder=1,
    )
    cbar = plt.colorbar(scatter_bg, ax=ax)
    cbar.set_label("Largest Singular Value of Jacobian", fontsize=10)

    # Plot ridge cells with specific state
    ridge_state_filtered = adata_on_ridge[
        adata_on_ridge.obs[state_key].isin([state_value])
    ]
    if len(ridge_state_filtered) > 0:
        ax.scatter(
            ridge_state_filtered.obsm["X_emb"][:, 0],
            ridge_state_filtered.obsm["X_emb"][:, 1],
            color=ridge_color,
            s=ridge_s,
            alpha=0.9,
            zorder=2,
            label=state_value,
        )

    ax.set_xlabel("Embedding Dim 1", fontsize=11)
    ax.set_ylabel("Embedding Dim 2", fontsize=11)
    ax.legend(loc="best", frameon=True, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
