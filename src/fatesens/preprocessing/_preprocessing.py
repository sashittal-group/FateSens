import scanpy as sc


def get_highly_variable_genes_subset(
    adata,
    normalized_counts_per_cell=1000,
    n_hvgs=500,
):
    adata_hvg = adata.copy()
    sc.pp.normalize_total(adata_hvg, target_sum=normalized_counts_per_cell)
    sc.pp.log1p(adata_hvg)
    sc.pp.highly_variable_genes(adata_hvg, n_top_genes=n_hvgs, subset=False)
    adata_hvg = adata_hvg[:, adata_hvg.var.highly_variable].copy()

    return adata_hvg