import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from anndata import AnnData

from UNAGI.plotting.manuscript_figures import (
    extract_dynamic_marker_table,
    extract_hierarchical_marker_table,
    plot_dynamic_marker_heatmap,
    plot_dynamic_marker_heatmap_track_panels,
    plot_dynamic_marker_trajectories,
    plot_hierarchical_marker_branch_heatmap,
    plot_perturbation_ranked_bar,
    plot_temporal_dynamics_graph,
)


def _build_mock_adata() -> AnnData:
    rng = np.random.default_rng(42)
    n_cells = 60
    genes = [f"GENE{i}" for i in range(8)]
    X = rng.normal(size=(n_cells, len(genes))).astype(float)

    stage = np.array(["0"] * 20 + ["1"] * 20 + ["2"] * 20)
    leiden = np.array(
        ["0"] * 10 + ["1"] * 10
        + ["2"] * 10 + ["3"] * 10
        + ["4"] * 10 + ["5"] * 10
    )
    ident_map = {"0": "AT1", "1": "AT2", "2": "Fib", "3": "Mono", "4": "Fib", "5": "Mono"}
    ident = np.array([ident_map[c] for c in leiden])

    adata = AnnData(
        X=X,
        obs=pd.DataFrame({"stage": stage, "leiden": leiden, "ident": ident}),
        var=pd.DataFrame(index=genes),
    )

    adata.uns["progressionMarkers"] = {
        "0-2-4": {
            "increasing": {
                "gene": {"0": "GENE0", "1": "GENE1", "2": "GENE2", "3": "AC_FAKE"},
                "rank": {"0": 1, "1": 2, "2": 3, "3": 4},
                "stage0": {"0": 0.10, "1": 0.05, "2": 0.02, "3": 0.03},
                "stage1": {"0": 0.32, "1": 0.18, "2": 0.10, "3": 0.09},
                "stage2": {"0": 0.62, "1": 0.35, "2": 0.20, "3": 0.11},
                "qval": {"0": 0.01, "1": 0.02, "2": 0.03, "3": 0.04},
            },
            "decreasing": {
                "gene": {"0": "GENE3", "1": "GENE4"},
                "rank": {"0": 1, "1": 2},
                "stage0": {"0": 0.42, "1": 0.30},
                "stage1": {"0": 0.22, "1": 0.18},
                "stage2": {"0": 0.10, "1": 0.07},
                "qval": {"0": 0.01, "1": 0.02},
            },
        }
    }

    pos_df = pd.DataFrame(
        {
            "pvals_adj": [0.001, 0.005, 0.03],
            "scores": [10.0, 8.0, 4.0],
            "log_fold_changes": [1.2, 0.8, 0.4],
            "Mean_compared_clusters": [0.2, 0.15, 0.1],
            "Mean_selected_clusters": [0.8, 0.7, 0.4],
        },
        index=["GENE0", "GENE1", "GENE2"],
    )
    neg_df = pd.DataFrame(
        {
            "pvals_adj": [0.002, 0.01],
            "scores": [9.0, 5.0],
            "log_fold_changes": [-1.1, -0.7],
            "Mean_compared_clusters": [0.8, 0.5],
            "Mean_selected_clusters": [0.2, 0.2],
        },
        index=["GENE3", "GENE4"],
    )
    adata.uns["hcmarkers"] = {
        "0": {
            "Z": np.array([[0.0, 1.0, 0.2, 2.0]]),
            "order": np.array([0, 1]),
            "markers": {
                "0": {
                    "0": {
                        "chosen": {"pos": pos_df, "neg": neg_df},
                        "ref": [[0], [0, 1]],
                    }
                }
            },
        }
    }

    adata.uns["edges"] = {
        "0": [[0, 2], [1, 3]],
        "1": [[2, 4], [3, 5]],
    }
    return adata


def test_dynamic_marker_plots_smoke():
    adata = _build_mock_adata()
    table = extract_dynamic_marker_table(adata, track="0-2-4", direction="increasing", top_n=2)
    assert not table.empty
    assert "stage0" in table.columns

    fig, _, out = plot_dynamic_marker_trajectories(
        adata,
        tracks=["0-2-4"],
        top_n=2,
        show=False,
    )
    assert not out.empty
    fig.clf()

    fig2, _, mat = plot_dynamic_marker_heatmap(
        adata,
        tracks=["0-2-4"],
        top_n=2,
        show=False,
    )
    assert not mat.empty
    fig2.clf()

    fig3, _, mat2 = plot_dynamic_marker_heatmap(
        adata,
        tracks=["0-2-4"],
        directions=("increasing", "decreasing"),
        top_n=3,
        per_track_total=True,
        show=False,
    )
    assert not mat2.empty
    assert len(mat2) <= 3
    fig3.clf()

    fig4, _, mat3 = plot_dynamic_marker_heatmap(
        adata,
        tracks=["0-2-4"],
        directions=("increasing",),
        top_n=5,
        per_track_total=True,
        protein_coding_only=True,
        require_genes_in_var=True,
        row_zscore=True,
        row_label_mode="track_gene",
        show=False,
    )
    assert not mat3.empty
    assert all("AC_FAKE" not in idx for idx in mat3.index.astype(str))
    fig4.clf()

    fig5, _, mat4 = plot_dynamic_marker_heatmap(
        adata,
        tracks=["0-2-4"],
        directions=("increasing", "decreasing"),
        top_n=3,
        per_track_total=True,
        cell_level=True,
        row_zscore=True,
        stage_key="stage",
        cluster_key="leiden",
        restrict_to_track_cells=True,
        show=False,
    )
    assert not mat4.empty
    assert mat4.shape[0] > 0
    assert mat4.shape[1] > 0
    all_expr = adata[:, mat4.columns.tolist()].X
    if hasattr(all_expr, "toarray"):
        all_expr = all_expr.toarray()
    else:
        all_expr = np.asarray(all_expr)
    gene_means = all_expr.mean(axis=0, keepdims=True)
    gene_stds = all_expr.std(axis=0, keepdims=True)
    gene_stds[gene_stds == 0.0] = 1.0

    subset_expr = adata[mat4.index.tolist(), mat4.columns.tolist()].X
    if hasattr(subset_expr, "toarray"):
        subset_expr = subset_expr.toarray()
    else:
        subset_expr = np.asarray(subset_expr)
    expected = (subset_expr - gene_means) / gene_stds
    assert np.allclose(mat4.to_numpy(dtype=float), expected, atol=1e-6)
    fig5.clf()

    fig6, _, panels = plot_dynamic_marker_heatmap_track_panels(
        adata,
        tracks=["0-2-4"],
        top_n=3,
        per_track_total=True,
        row_zscore=True,
        stage_key="stage",
        cluster_key="leiden",
        cell_groupby="stage",
        show=False,
    )
    assert not panels.empty
    assert "track" in panels.columns
    fig6.clf()

def test_hierarchical_marker_plot_smoke():
    adata = _build_mock_adata()
    table = extract_hierarchical_marker_table(
        adata,
        stage=0,
        cluster=0,
        level=0,
        direction="up",
        top_n=2,
    )
    assert not table.empty
    assert {"gene", "log_fold_changes"}.issubset(table.columns)

    fig, _, matrix, markers = plot_hierarchical_marker_branch_heatmap(
        adata,
        stage=0,
        cluster=0,
        level=0,
        direction="up",
        top_n=2,
        show=False,
    )
    assert not matrix.empty
    assert not markers.empty
    fig.clf()


def test_temporal_dynamics_graph_smoke():
    adata = _build_mock_adata()
    fig, _, nodes, edges = plot_temporal_dynamics_graph(adata, show=False)
    assert not nodes.empty
    assert not edges.empty
    assert {"src_stage", "dst_stage"}.issubset(edges.columns)
    fig.clf()


def test_perturbation_ranked_bar_wraps_long_labels():
    table = pd.DataFrame(
        {
            "pathways": [
                "HALLMARK_TGF_BETA_SIGNALING_VIA_SMAD_AND_NON_CANONICAL_BRANCHES",
                "REACTOME_SIGNALING_BY_TGF_BETA_RECEPTOR_COMPLEX",
            ],
            "pval_adjusted": [1e-4, 2e-3],
            "perturbation score": [0.8, 0.6],
        }
    )

    fig, ax, out = plot_perturbation_ranked_bar(table, show=False)

    tick_texts = [tick.get_text() for tick in ax.get_yticklabels()]
    assert not out.empty
    assert ax.get_ylabel() == "Pathways"
    assert all(text.count("\n") <= 1 for text in tick_texts)
    assert all("…" not in text for text in tick_texts)
    fig.clf()
