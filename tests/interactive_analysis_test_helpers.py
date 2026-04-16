from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData


def build_mock_analysis_adata(
    *,
    include_hierarchical: bool = True,
    include_dynamic: bool = True,
    include_perturbation: bool = True,
    include_umap: bool = True,
    stage_labels: tuple[str, ...] = ("0", "1", "2"),
    cells_per_stage: int = 24,
) -> AnnData:
    rng = np.random.default_rng(42)
    genes = [f"GENE{i}" for i in range(10)]
    total_cells = len(stage_labels) * cells_per_stage
    X = rng.normal(size=(total_cells, len(genes))).astype(float)

    stage = np.repeat(np.array(stage_labels, dtype=str), cells_per_stage)
    base_leiden = np.array(["0", "1", "2", "3", "4", "5"], dtype=str)
    leiden = np.resize(base_leiden, total_cells)
    ident_map = {"0": "AT1", "1": "AT2", "2": "Fib", "3": "Mono", "4": "Fib", "5": "Mono"}
    ident = np.array([ident_map[item] for item in leiden])

    adata = AnnData(
        X=X,
        obs=pd.DataFrame({"stage": stage, "leiden": leiden, "ident": ident}),
        var=pd.DataFrame(index=genes),
    )
    if include_umap:
        adata.obsm["X_umap"] = rng.normal(size=(total_cells, 2)).astype(float)

    if include_dynamic:
        adata.uns["progressionMarkers"] = {
            "0-2-4": {
                "increasing": {
                    "gene": {"0": "GENE0", "1": "GENE1", "2": "GENE2"},
                    "rank": {"0": 1, "1": 2, "2": 3},
                    "stage0": {"0": 0.1, "1": 0.2, "2": 0.3},
                    "stage1": {"0": 0.4, "1": 0.5, "2": 0.6},
                    "stage2": {"0": 0.7, "1": 0.8, "2": 0.9},
                },
                "decreasing": {
                    "gene": {"0": "GENE3", "1": "GENE4"},
                    "rank": {"0": 1, "1": 2},
                    "stage0": {"0": 0.9, "1": 0.8},
                    "stage1": {"0": 0.6, "1": 0.5},
                    "stage2": {"0": 0.4, "1": 0.3},
                },
            }
        }
        adata.uns["edges"] = {
            "0": [["0", "2"]],
            "1": [["2", "4"]],
        }

    if include_hierarchical:
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

    if include_perturbation:
        adata.uns["data_pathway_overlap_genes"] = {
            "PathA": ["GENE0", "GENE1"],
            "PathB": ["GENE2", "GENE3"],
            "PathTrack": ["GENE0", "GENE4"],
        }
        adata.uns["data_drug_overlap_genes"] = {
            "DrugA": ["GENE0:+", "GENE1:-"],
            "DrugB": ["GENE2:+", "GENE3:-"],
        }
        adata.uns["data_two_drug_overlap_genes"] = {
            "DrugA__DrugB": ["GENE0:+", "GENE1:-", "GENE2:+", "GENE3:-"],
        }
        for key, values in {
            "pathway_perturbation_score": {
                "0.5": {
                    "overall": {
                        "top_compounds": {
                            "compound": ["PathA", "PathB"],
                            "perturbation score": [3.2, 2.4],
                            "pval_adjusted": [0.001, 0.02],
                        }
                    },
                    "0": {
                        "top_compounds": {
                            "compound": ["PathTrack"],
                            "perturbation score": [2.1],
                            "pval_adjusted": [0.01],
                        }
                    },
                }
            },
            "drug_perturbation_score": {
                "0.5": {
                    "overall": {
                        "top_compounds": {
                            "compound": ["DrugA", "DrugB"],
                            "perturbation score": [4.5, 3.6],
                            "pval_adjusted": [0.0005, 0.04],
                        }
                    }
                }
            },
            "single_gene_perturbation_score": {
                "0.5": {
                    "overall": {
                        "top_compounds": {
                            "compound": ["GENE0", "GENE1"],
                            "perturbation score": [5.1, 4.4],
                            "pval_adjusted": [0.0002, 0.03],
                        }
                    }
                }
            },
            "two_genes_perturbation_score": {
                "0.5": {
                    "overall": {
                        "top_compounds": {
                            "compound": ["GENE0_GENE1", "GENE2_GENE3"],
                            "perturbation score": [6.0, 4.8],
                            "pval_adjusted": [0.0001, 0.02],
                        }
                    }
                }
            },
            "two_drugs_perturbation_score": {
                "0.5": {
                    "overall": {
                        "top_compounds": {
                            "compound": ["DrugA__DrugB"],
                            "perturbation score": [4.9],
                            "pval_adjusted": [0.004],
                        }
                    }
                }
            },
        }.items():
            adata.uns[key] = values

    return adata


def write_analysis_dataset(
    result_dir: Path,
    *,
    include_hierarchical: bool = True,
    include_dynamic: bool = True,
    include_perturbation: bool = True,
    include_stage_specific_perturbation: bool = False,
    include_umap: bool = True,
    stage_labels: tuple[str, ...] = ("0", "1", "2"),
    cells_per_stage: int = 24,
) -> Path:
    result_dir.mkdir(parents=True, exist_ok=True)
    adata = build_mock_analysis_adata(
        include_hierarchical=include_hierarchical,
        include_dynamic=include_dynamic,
        include_perturbation=include_perturbation,
        include_umap=include_umap,
        stage_labels=stage_labels,
        cells_per_stage=cells_per_stage,
    )
    uns_payload = dict(adata.uns)
    adata.uns = {}
    adata.write_h5ad(result_dir / "dataset.h5ad")
    with (result_dir / "attribute.pkl").open("wb") as handle:
        pickle.dump(uns_payload, handle)
    if include_stage_specific_perturbation:
        stage_root = result_dir / "stage_specific_perturbation"
        for stage in stage_labels:
            stage_dir = stage_root / f"stage_{stage}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            pathway_frame = pd.DataFrame(
                {
                    "rank": [0, 1],
                    "pathways": [f"PathwayStage{stage}A", f"PathwayStage{stage}B"],
                    "perturbation score": [3.5 - 0.1 * int(stage), 2.2 - 0.1 * int(stage)],
                    "pval_adjusted": [0.001, 0.02],
                    "regulated genes": ["['GENE0']", "['GENE1']"],
                    "idrem_suggestion": ["['GENE0:+']", "['GENE1:-']"],
                }
            )
            drug_frame = pd.DataFrame(
                {
                    "rank": [0, 1],
                    "compound": [f"DrugStage{stage}A", f"DrugStage{stage}B"],
                    "perturbation score": [4.0 - 0.1 * int(stage), 3.1 - 0.1 * int(stage)],
                    "pval_adjusted": [0.0005, 0.04],
                    "regulated genes": ["['GENE2:+']", "['GENE3:-']"],
                    "idrem_suggestion": ["['GENE2:+']", "['GENE3:-']"],
                }
            )
            single_gene_frame = pd.DataFrame(
                {
                    "rank": [0, 1],
                    "compound": [f"GENE_STAGE_{stage}_A", f"GENE_STAGE_{stage}_B"],
                    "perturbation score": [4.4 - 0.1 * int(stage), 3.0 - 0.1 * int(stage)],
                    "pval_adjusted": [0.002, 0.03],
                }
            )
            gene_combo_frame = pd.DataFrame(
                {
                    "rank": [0, 1],
                    "compound": [f"GENE_A_GENE_B_STAGE_{stage}", f"GENE_C_GENE_D_STAGE_{stage}"],
                    "perturbation score": [5.4 - 0.1 * int(stage), 4.1 - 0.1 * int(stage)],
                    "pval_adjusted": [0.001, 0.02],
                }
            )
            two_drug_frame = pd.DataFrame(
                {
                    "rank": [0, 1],
                    "compound": [f"DrugStage{stage}A__DrugStage{stage}B", f"DrugStage{stage}C__DrugStage{stage}D"],
                    "perturbation score": [4.7 - 0.1 * int(stage), 3.4 - 0.1 * int(stage)],
                    "pval_adjusted": [0.003, 0.04],
                }
            )
            pathway_frame.to_csv(stage_dir / f"pathway_perturbation_overall_stage_{stage}.csv", index=False)
            drug_frame.to_csv(stage_dir / f"drug_perturbation_overall_stage_{stage}.csv", index=False)
            single_gene_frame.to_csv(stage_dir / f"single_gene_perturbation_overall_stage_{stage}.csv", index=False)
            gene_combo_frame.to_csv(stage_dir / f"gene_combo_perturbation_overall_stage_{stage}.csv", index=False)
            two_drug_frame.to_csv(stage_dir / f"two_drugs_perturbation_overall_stage_{stage}.csv", index=False)
    return result_dir
