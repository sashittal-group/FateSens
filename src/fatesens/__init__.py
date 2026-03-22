from . import pp, flow_map, tl, jacobian_matrix


import anndata

# Ensure compatibility between modern AnnData and older WOT calls
if not hasattr(anndata, "read"):
    anndata.read = anndata.read_h5ad


def hello() -> str:
    return "Hello from fatesens!"
