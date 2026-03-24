from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt

from UNAGI.plotting import (
    plot_dynamic_marker_heatmap,
    plot_dynamic_marker_trajectories,
    plot_hierarchical_marker_branch_heatmap,
    plot_perturbation_ranked_bar,
    plot_stage_cell_type_composition,
    plot_stage_umap_grid,
    plot_temporal_dynamics_graph,
)

from .visualization_pipeline import (
    _first_hc_branch,
    _first_progression_track,
    _read_analysis_adata,
    _resolve_perturb_change,
)

STATIC_PDF_EXPORTS = {
    "stage_umap_grid",
    "stage_composition",
    "pathway_ranked_bar",
    "drug_ranked_bar",
    "single_gene_ranked_bar",
    "gene_combo_ranked_bar",
    "dynamic_marker_trajectories",
    "dynamic_marker_heatmap",
    "hierarchical_marker_heatmap",
    "temporal_dynamics_graph",
}


def _figure_to_pdf_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def export_static_publication_pdf(analysis_dir: str | Path, figure_name: str) -> tuple[bytes, str]:
    analysis_dir = Path(analysis_dir).expanduser().resolve()
    adata, _dataset_path, _attribute_path = _read_analysis_adata(analysis_dir)
    name = str(figure_name).strip()

    if name == "stage_umap_grid":
        fig, _axes = plot_stage_umap_grid(adata, show=False)
        return _figure_to_pdf_bytes(fig), "stage_umap_grid_publication.pdf"

    if name == "stage_composition":
        fig, _ax, _ctab = plot_stage_cell_type_composition(adata, show=False)
        return _figure_to_pdf_bytes(fig), "stage_composition_publication.pdf"

    perturbation_modes = {
        "pathway_ranked_bar": ("pathway", "pathway_perturbation_score"),
        "drug_ranked_bar": ("drug", "drug_perturbation_score"),
        "single_gene_ranked_bar": ("single_gene", "single_gene_perturbation_score"),
        "gene_combo_ranked_bar": ("gene_combo", "two_genes_perturbation_score"),
    }
    if name in perturbation_modes:
        mode, uns_key = perturbation_modes[name]
        perturb_change = _resolve_perturb_change(adata, uns_key)
        from UNAGI.plotting import extract_perturbation_table

        table = extract_perturbation_table(
            adata,
            mode=mode,
            perturb_change=perturb_change,
            track="overall",
        )
        use_pval = mode != "pathway"
        fig, _ax, _work = plot_perturbation_ranked_bar(table, use_neglog10_pval=use_pval, show=False)
        return _figure_to_pdf_bytes(fig), f"{name}_publication.pdf"

    if name == "dynamic_marker_trajectories":
        track = _first_progression_track(adata)
        fig, _axes, _table = plot_dynamic_marker_trajectories(
            adata,
            tracks=[track],
            directions=("increasing",),
            show=False,
        )
        return _figure_to_pdf_bytes(fig), "dynamic_marker_trajectories_publication.pdf"

    if name == "dynamic_marker_heatmap":
        track = _first_progression_track(adata)
        fig, _axes, _table = plot_dynamic_marker_heatmap(
            adata,
            tracks=[track],
            directions=("increasing",),
            show=False,
        )
        return _figure_to_pdf_bytes(fig), "dynamic_marker_heatmap_publication.pdf"

    if name == "hierarchical_marker_heatmap":
        stage, cluster, level = _first_hc_branch(adata)
        fig, _ax, _matrix, _markers = plot_hierarchical_marker_branch_heatmap(
            adata,
            stage=stage,
            cluster=cluster,
            level=level,
            direction="up",
            show=False,
        )
        return _figure_to_pdf_bytes(fig), "hierarchical_marker_heatmap_publication.pdf"

    if name == "temporal_dynamics_graph":
        fig, _ax, _nodes, _edges = plot_temporal_dynamics_graph(adata, show=False)
        return _figure_to_pdf_bytes(fig), "temporal_dynamics_graph_publication.pdf"

    raise KeyError(f"Unsupported static figure export: {name}")
