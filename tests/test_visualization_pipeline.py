from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from anndata import AnnData

from UNAGI.monitoring.visualization_pipeline import generate_analysis_visualizations

matplotlib.use("Agg")


def _build_mock_adata() -> AnnData:
    rng = np.random.default_rng(42)
    genes = [f"GENE{i}" for i in range(8)]
    X = rng.normal(size=(60, len(genes))).astype(float)

    stage = np.array(["0"] * 20 + ["1"] * 20 + ["2"] * 20)
    leiden = np.array(["0"] * 10 + ["1"] * 10 + ["2"] * 10 + ["3"] * 10 + ["4"] * 10 + ["5"] * 10)
    ident_map = {"0": "AT1", "1": "AT2", "2": "Fib", "3": "Mono", "4": "Fib", "5": "Mono"}
    ident = np.array([ident_map[c] for c in leiden])

    adata = AnnData(
        X=X,
        obs=pd.DataFrame({"stage": stage, "leiden": leiden, "ident": ident}),
        var=pd.DataFrame(index=genes),
    )
    adata.obsm["X_umap"] = rng.normal(size=(60, 2)).astype(float)

    adata.uns["progressionMarkers"] = {
        "0-2-4": {
            "increasing": {
                "gene": {"0": "GENE0", "1": "GENE1", "2": "GENE2"},
                "rank": {"0": 1, "1": 2, "2": 3},
                "stage0": {"0": 0.1, "1": 0.2, "2": 0.3},
                "stage1": {"0": 0.4, "1": 0.5, "2": 0.6},
                "stage2": {"0": 0.7, "1": 0.8, "2": 0.9},
            }
        }
    }

    pos_df = pd.DataFrame(
        {
            "pvals_adj": [0.001, 0.005],
            "scores": [10.0, 8.0],
            "log_fold_changes": [1.2, 0.8],
            "Mean_compared_clusters": [0.2, 0.15],
            "Mean_selected_clusters": [0.8, 0.7],
        },
        index=["GENE0", "GENE1"],
    )
    neg_df = pd.DataFrame(
        {
            "pvals_adj": [0.002],
            "scores": [9.0],
            "log_fold_changes": [-1.1],
            "Mean_compared_clusters": [0.8],
            "Mean_selected_clusters": [0.2],
        },
        index=["GENE3"],
    )
    adata.uns["hcmarkers"] = {
        "0": {
            "Z": np.array([[0.0, 1.0, 0.2, 2.0]]),
            "order": np.array([0, 1]),
            "markers": {"0": {"0": {"chosen": {"pos": pos_df, "neg": neg_df}, "ref": [[0], [0, 1]]}}},
        }
    }
    adata.uns["edges"] = {"0": [[0, 2], [1, 3]], "1": [[2, 4], [3, 5]]}
    return adata


def test_generate_analysis_visualizations_writes_manifest_and_handles_missing_optional_outputs(tmp_path):
    analysis_dir = tmp_path / "analysis" / "iter1"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    adata = _build_mock_adata()
    uns_payload = dict(adata.uns)
    adata.uns = {}
    adata.write_h5ad(analysis_dir / "dataset.h5ad")
    with (analysis_dir / "attribute.pkl").open("wb") as handle:
        pickle.dump(uns_payload, handle)

    manifest = generate_analysis_visualizations(analysis_dir)

    manifest_path = analysis_dir / "visualizations" / "manifest.json"
    assert manifest_path.exists()
    assert manifest["counts"]["successes"] > 0
    success_names = {entry["name"] for entry in manifest["successes"]}
    assert "stage_umap_grid" in success_names
    assert "dynamic_marker_trajectories" in success_names
    skipped_names = {entry["name"] for entry in manifest["skipped"]}
    assert "single_gene_perturbation" in skipped_names
    assert "gene_combo_perturbation" in skipped_names
    assert manifest["counts"]["skipped"] >= 2
