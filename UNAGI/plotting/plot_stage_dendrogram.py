from __future__ import annotations

from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch


def _sorted_stage_labels(values: Iterable[Any]) -> list[Any]:
    vals = list(values)
    try:
        return sorted(vals, key=lambda x: float(str(x)))
    except (TypeError, ValueError):
        return sorted(vals, key=lambda x: str(x))


def plot_hc_dendrogram(adata, stage_key, celltype_key, save=False, dpi=None):
    """
    Plot hierarchical-marker dendrograms for all stages.

    This is a cleaned replacement of the legacy implementation with the same API.
    """
    if "hcmarkers" not in adata.uns:
        raise KeyError("Missing adata.uns['hcmarkers']. Run downstream analysis first.")
    if stage_key not in adata.obs:
        raise KeyError(f"Missing stage key in adata.obs: {stage_key}")
    if celltype_key not in adata.obs:
        raise KeyError(f"Missing cell type key in adata.obs: {celltype_key}")

    stages = _sorted_stage_labels(adata.obs[stage_key].astype(str).unique().tolist())
    for stage in stages:
        stage_s = str(stage)
        if stage_s not in adata.uns["hcmarkers"]:
            continue
        Z = adata.uns["hcmarkers"][stage_s]["Z"]
        leaves = sch.leaves_list(Z)
        stage_adata = adata[adata.obs[stage_key].astype(str) == stage_s].to_memory() if adata.isbacked else adata[adata.obs[stage_key].astype(str) == stage_s].copy()
        stage_adata.obs["leiden"] = stage_adata.obs["leiden"].astype(str)
        stage_adata.obs["_leiden_celltype"] = stage_adata.obs["leiden"] + "_" + stage_adata.obs[celltype_key].astype(str)

        leaf_labels = []
        for leaf in leaves:
            leaf_str = str(leaf)
            sub = stage_adata[stage_adata.obs["leiden"] == leaf_str]
            if sub.n_obs == 0:
                leaf_labels.append(leaf_str)
            else:
                leaf_labels.append(sub.obs["_leiden_celltype"].iloc[0])

        fig, ax = plt.subplots(figsize=(10, 10), dpi=(dpi or 120))
        sch.dendrogram(Z, ax=ax)
        ax.set_xticklabels(leaf_labels, rotation=90)
        ax.set_title(f"{stage_key}={stage_s}")
        fig.tight_layout()
        if save:
            fig.savefig(f"dendrogram_stage_{stage_s}.pdf", dpi=(dpi or 300))
        plt.show()


__all__ = ["plot_hc_dendrogram"]

