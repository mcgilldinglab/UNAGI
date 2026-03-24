"""
The plotting module contains functions for visualizing UNAGI outcomes.
"""

from .manuscript_figures import (
    build_palette,
    extract_dynamic_marker_table,
    extract_hierarchical_marker_table,
    extract_perturbation_table,
    plot_dynamic_marker_heatmap,
    plot_dynamic_marker_heatmap_track_panels,
    plot_dynamic_marker_trajectories,
    plot_distribution_overlap,
    plot_dynamic_trajectory_graph,
    plot_gene_expression_shift,
    plot_hierarchical_marker_branch_heatmap,
    plot_insilico_vs_observed_scatter,
    plot_marker_dotplot,
    plot_perturbation_ranked_bar,
    plot_stage_cell_type_composition,
    plot_stage_umap_grid,
    plot_temporal_dynamics_graph,
)
from .plot_cell_embeddings import plot_stages_latent_representation, plot_with_colormap
from .plot_cell_type_composition import cell_type_composition
from .plot_hierarchical_markers_heatmap import hierarchical_static_markers_heatmap
from .plot_stage_dendrogram import plot_hc_dendrogram

__all__ = [
    "build_palette",
    "plot_stage_umap_grid",
    "plot_stage_cell_type_composition",
    "plot_marker_dotplot",
    "plot_dynamic_trajectory_graph",
    "extract_dynamic_marker_table",
    "plot_dynamic_marker_trajectories",
    "plot_dynamic_marker_heatmap",
    "plot_dynamic_marker_heatmap_track_panels",
    "extract_hierarchical_marker_table",
    "plot_hierarchical_marker_branch_heatmap",
    "plot_temporal_dynamics_graph",
    "extract_perturbation_table",
    "plot_perturbation_ranked_bar",
    "plot_gene_expression_shift",
    "plot_insilico_vs_observed_scatter",
    "plot_distribution_overlap",
    "plot_with_colormap",
    "plot_stages_latent_representation",
    "cell_type_composition",
    "hierarchical_static_markers_heatmap",
    "plot_hc_dendrogram",
]
