from __future__ import annotations

import os
from pathlib import Path


def _prepare_scanpy():
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
    import scanpy as sc

    return sc


def hierarchical_static_markers_heatmap(
    adata,
    stage,
    cluster,
    level,
    n_genes,
    stage_key,
    celltype_key,
    protein_encoding_gene=True,
    min_logfoldchange=1,
    dpi=None,
    save=None,
):
    """
    Plot a hierarchical-marker heatmap for one stage/cluster branch.
    """
    sc = _prepare_scanpy()
    if "hcmarkers" not in adata.uns:
        raise KeyError("Missing adata.uns['hcmarkers']. Run downstream analysis first.")
    stage_s = str(stage)
    if stage_s not in adata.uns["hcmarkers"]:
        raise KeyError(f"Stage {stage_s} not found in adata.uns['hcmarkers']")
    if stage_key not in adata.obs:
        raise KeyError(f"Missing stage key in adata.obs: {stage_key}")
    if celltype_key not in adata.obs:
        raise KeyError(f"Missing cell type key in adata.obs: {celltype_key}")

    if protein_encoding_gene:
        keep = [
            g
            for g in adata.var_names
            if not (
                g.startswith("AC")
                or g.startswith("AL")
                or g.startswith("AP")
                or g.startswith("LINC")
                or "." in g
            )
        ]
        adata = adata[:, keep].to_memory() if adata.isbacked else adata[:, keep].copy()

    sc.set_figure_params(scanpy=True, dpi=80, dpi_save=(dpi or 300))
    stage_adata = adata[adata.obs[stage_key].astype(str) == stage_s].to_memory() if adata.isbacked else adata[adata.obs[stage_key].astype(str) == stage_s].copy()
    if stage_adata.n_obs == 0:
        raise ValueError(f"No cells found for {stage_key}={stage_s}")

    marker_meta = stage_adata.uns["hcmarkers"][stage_s]["markers"][str(cluster)][str(level)]
    chosen = marker_meta["ref"][0]
    sibling_scope = [x for x in marker_meta["ref"][1] if x not in chosen]

    stage_adata.obs["leiden"] = stage_adata.obs["leiden"].astype(str)
    chosen_adata = stage_adata[stage_adata.obs["leiden"].isin([str(v) for v in chosen])].to_memory() if stage_adata.isbacked else stage_adata[stage_adata.obs["leiden"].isin([str(v) for v in chosen])].copy()
    siblings_adata = stage_adata[stage_adata.obs["leiden"].isin([str(v) for v in sibling_scope])].to_memory() if stage_adata.isbacked else stage_adata[stage_adata.obs["leiden"].isin([str(v) for v in sibling_scope])].copy()
    chosen_adata.obs["_subset"] = "chosen"
    siblings_adata.obs["_subset"] = "siblings"
    compare_adata = chosen_adata.concatenate(siblings_adata)
    compare_adata.obs["_subset"] = compare_adata.obs["_subset"].astype("category")
    compare_adata.layers["scaled"] = sc.pp.scale(compare_adata, copy=True).X
    sc.tl.rank_genes_groups(compare_adata, "_subset", method="wilcoxon", n_genes=n_genes)

    marker_genes = list(compare_adata.uns["rank_genes_groups"]["names"]["chosen"])[:n_genes]
    if save is not None:
        sc.pl.heatmap(
            compare_adata,
            marker_genes,
            groupby=["_subset", celltype_key, "leiden"],
            layer="scaled",
            swap_axes=True,
            vmin=-0.5,
            vmax=0.5,
            cmap="RdBu_r",
            figsize=(20, 16),
            save=save,
            show=False,
        )
    else:
        sc.pl.heatmap(
            compare_adata,
            marker_genes,
            groupby=["_subset", celltype_key, "leiden"],
            layer="scaled",
            swap_axes=True,
            vmin=-0.5,
            vmax=0.5,
            cmap="RdBu_r",
            figsize=(20, 16),
            show=True,
        )


__all__ = ["hierarchical_static_markers_heatmap"]

