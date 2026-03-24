from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from .manuscript_figures import build_palette, plot_stage_umap_grid


def plot_with_colormap(values, color_dict):
    """
    Backward-compatible helper to assign colors to category labels.
    """
    merged = dict(color_dict) if color_dict is not None else {}
    return build_palette(values, preset=merged)


def plot_stages_latent_representation(
    adatas,
    cell_type_key,
    stage_key,
    color_scheme: Optional[Mapping[Any, str]] = None,
    ax=None,
    dpi: int = 300,
    save: Optional[str] = None,
):
    """
    Backward-compatible wrapper for stage-wise UMAP panels.

    It now uses `plot_stage_umap_grid` under the hood and additionally reports
    ARI/NMI/silhouette when compatible columns exist.
    """
    if ax is not None:
        print("Warning: `ax` argument is ignored in this wrapper; using an internal subplot grid.")

    fig, axes = plot_stage_umap_grid(
        adatas,
        stage_key=stage_key,
        color_keys=(cell_type_key, "leiden"),
        palette=color_scheme,
        dpi=dpi,
        save=save,
        show=save is None,
    )

    # Optional metric summary for compatibility with prior behavior.
    if {"name.simple", "leiden"}.issubset(adatas.obs.columns) and "z" in adatas.obsm:
        stage_vals = adatas.obs[stage_key].astype(str).unique().tolist()
        stage_vals = sorted(stage_vals, key=lambda x: float(x) if str(x).replace(".", "", 1).isdigit() else str(x))
        ari_vals = []
        nmi_vals = []
        sil_vals = []
        for stage in stage_vals:
            sub = adatas[adatas.obs[stage_key].astype(str) == str(stage)]
            if sub.n_obs < 3:
                continue
            try:
                ari_vals.append(adjusted_rand_score(sub.obs["name.simple"], sub.obs[cell_type_key].astype(str)))
                nmi_vals.append(normalized_mutual_info_score(sub.obs["name.simple"], sub.obs[cell_type_key].astype(str)))
            except Exception:
                pass
            try:
                sil_vals.append(silhouette_score(sub.obsm["z"], sub.obs["leiden"].astype(str)))
            except Exception:
                pass
        if ari_vals:
            print("ARIs:", ari_vals)
            print("Mean ARI:", sum(ari_vals) / len(ari_vals))
        if nmi_vals:
            print("NMIs:", nmi_vals)
            print("Mean NMI:", sum(nmi_vals) / len(nmi_vals))
        if sil_vals:
            print("Silhouette scores:", sil_vals)
            print("Mean silhouette:", sum(sil_vals) / len(sil_vals))

    return fig, axes


__all__ = ["plot_with_colormap", "plot_stages_latent_representation"]
