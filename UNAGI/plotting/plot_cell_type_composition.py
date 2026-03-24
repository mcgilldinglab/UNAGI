from __future__ import annotations

from typing import Any, Mapping, Optional

from .manuscript_figures import plot_stage_cell_type_composition


def cell_type_composition(
    adata,
    cell_type_key,
    stage_key,
    ax=None,
    dpi: int = 300,
    show_cutoff: float = 0.04,
    colormaps: str = "Spectral",
    category_colors=None,
    save: Optional[str] = None,
):
    """
    Backward-compatible wrapper for stage-wise stacked composition bar plots.
    """
    _ = colormaps  # kept for compatibility
    palette: Optional[Mapping[Any, str]] = None
    if category_colors is not None:
        cell_types = sorted(adata.obs[cell_type_key].astype(str).unique().tolist())
        palette = {k: category_colors[i % len(category_colors)] for i, k in enumerate(cell_types)}

    fig, ax_out, _ctab = plot_stage_cell_type_composition(
        adata,
        cell_type_key=cell_type_key,
        stage_key=stage_key,
        min_label_fraction=show_cutoff,
        palette=palette,
        dpi=dpi,
        ax=ax,
        save=save,
        show=save is None,
    )
    return fig, ax_out


__all__ = ["cell_type_composition"]

