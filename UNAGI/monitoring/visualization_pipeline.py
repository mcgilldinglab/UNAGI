from __future__ import annotations

import json
import os
import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import scipy.cluster.hierarchy as sch

from UNAGI.plotting import (
    extract_dynamic_marker_table,
    extract_hierarchical_marker_table,
    extract_perturbation_table,
    plot_dynamic_marker_heatmap,
    plot_dynamic_marker_heatmap_track_panels,
    plot_dynamic_marker_trajectories,
    plot_hierarchical_marker_branch_heatmap,
    plot_perturbation_ranked_bar,
    plot_stage_cell_type_composition,
    plot_stage_umap_grid,
    plot_temporal_dynamics_graph,
)

INTERACTIVE_UNS_KEYS = (
    'edges',
    'progressionMarkers',
    'hcmarkers',
    'pathway_perturbation_score',
    'drug_perturbation_score',
    'single_gene_perturbation_score',
    'two_genes_perturbation_score',
    'two_drugs_perturbation_score',
    'data_pathway_overlap_genes',
    'data_drug_overlap_genes',
    'data_two_drug_overlap_genes',
)


def _sorted_keys(values: Iterable[Any]) -> list[str]:
    def sort_key(value: Any) -> tuple[int, Any]:
        text = str(value)
        try:
            return (0, float(text))
        except ValueError:
            return (1, text)

    return [str(v) for v in sorted(values, key=sort_key)]


def _filename_token(value: Any, *, fallback: str = "item") -> str:
    text = str(value).strip()
    if not text:
        return fallback
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    while "__" in text:
        text = text.replace("__", "_")
    text = text.strip("._-")
    return text or fallback


def _write_csv(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return str(path)


def _close_fig(fig) -> None:
    if fig is not None:
        plt.close(fig)


def _file_fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    stat = path.stat()
    return {
        'path': str(path.resolve()),
        'mtime_ns': int(stat.st_mtime_ns),
        'size': int(stat.st_size),
    }


def _interactive_uns_cache_paths(analysis_dir: Path) -> tuple[Path, Path]:
    return (
        analysis_dir / 'interactive_uns_subset.pkl',
        analysis_dir / 'interactive_uns_subset_manifest.json',
    )


def _load_analysis_uns_subset(
    analysis_dir: Path,
    attribute_path: Path | None,
    *,
    uns_keys: Iterable[str],
) -> dict[str, Any]:
    if attribute_path is None or not attribute_path.exists():
        return {}
    requested_keys = tuple(sorted(str(key) for key in uns_keys))
    subset_path, manifest_path = _interactive_uns_cache_paths(analysis_dir)
    expected_manifest = {
        'dataset': _file_fingerprint(analysis_dir / 'dataset.h5ad'),
        'attribute': _file_fingerprint(attribute_path),
        'keys': list(requested_keys),
    }

    if subset_path.exists() and manifest_path.exists():
        try:
            with manifest_path.open('r', encoding='utf-8') as handle:
                cached_manifest = json.load(handle)
            if cached_manifest == expected_manifest:
                with subset_path.open('rb') as handle:
                    cached_uns = pickle.load(handle)
                if isinstance(cached_uns, Mapping):
                    return dict(cached_uns)
        except Exception:
            pass

    with attribute_path.open('rb') as handle:
        loaded_uns = pickle.load(handle)
    if not isinstance(loaded_uns, Mapping):
        return {}
    subset_uns = {key: loaded_uns[key] for key in requested_keys if key in loaded_uns}
    try:
        with subset_path.open('wb') as handle:
            pickle.dump(subset_uns, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with manifest_path.open('w', encoding='utf-8') as handle:
            json.dump(expected_manifest, handle, indent=2)
    except OSError:
        pass
    return subset_uns


def _read_analysis_adata(
    analysis_dir: Path,
    *,
    backed: bool = False,
    uns_keys: Iterable[str] | None = None,
):
    dataset_path = analysis_dir / 'dataset.h5ad'
    if not dataset_path.exists():
        raise FileNotFoundError(f'Analysis dataset not found: {dataset_path}')
    adata = sc.read_h5ad(dataset_path, backed='r' if backed else None)

    attribute_path = analysis_dir / 'attribute.pkl'
    loaded_uns: Any = None
    if attribute_path.exists():
        if uns_keys is None:
            with attribute_path.open('rb') as handle:
                loaded_uns = pickle.load(handle)
        else:
            loaded_uns = _load_analysis_uns_subset(analysis_dir, attribute_path, uns_keys=uns_keys)
    if isinstance(loaded_uns, Mapping):
        adata.uns.update(dict(loaded_uns))
    elif loaded_uns is not None:
        adata.uns = loaded_uns
    return adata, dataset_path, attribute_path if attribute_path.exists() else None


def _first_progression_track(adata) -> str:
    progression = adata.uns.get("progressionMarkers")
    if not isinstance(progression, dict) or not progression:
        raise KeyError("Missing progressionMarkers in adata.uns")
    return _sorted_keys(progression.keys())[0]


def _first_hc_branch(adata) -> tuple[str, str, str]:
    hcmarkers = adata.uns.get("hcmarkers")
    if not isinstance(hcmarkers, dict) or not hcmarkers:
        raise KeyError("Missing hcmarkers in adata.uns")

    for stage in _sorted_keys(hcmarkers.keys()):
        markers = hcmarkers[stage].get("markers", {})
        if not isinstance(markers, dict) or not markers:
            continue
        for cluster in _sorted_keys(markers.keys()):
            levels = markers[cluster]
            if not isinstance(levels, dict) or not levels:
                continue
            for level in _sorted_keys(levels.keys()):
                return stage, cluster, level
    raise ValueError("No hierarchical marker branches available")


def _resolve_perturb_change(adata, uns_key: str) -> str:
    bucket = adata.uns.get(uns_key)
    if not isinstance(bucket, dict) or not bucket:
        raise KeyError(f"Missing {uns_key} in adata.uns")
    return _sorted_keys(bucket.keys())[0]


@contextmanager
def _chdir(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _generate_hc_dendrograms(adata, figure_dir: Path) -> list[str]:
    if "hcmarkers" not in adata.uns:
        raise KeyError("Missing hcmarkers in adata.uns")
    if "stage" not in adata.obs or "ident" not in adata.obs:
        raise KeyError("Dendrogram generation requires adata.obs['stage'] and adata.obs['ident']")

    out_files: list[str] = []
    for stage in _sorted_keys(adata.obs["stage"].astype(str).unique().tolist()):
        stage_bucket = adata.uns["hcmarkers"].get(str(stage))
        if not isinstance(stage_bucket, dict) or "Z" not in stage_bucket:
            continue
        Z = stage_bucket["Z"]
        leaves = sch.leaves_list(Z)
        stage_adata = adata[adata.obs["stage"].astype(str) == str(stage)].to_memory() if adata.isbacked else adata[adata.obs["stage"].astype(str) == str(stage)].copy()
        stage_adata.obs["leiden"] = stage_adata.obs["leiden"].astype(str)
        stage_adata.obs["_leiden_celltype"] = (
            stage_adata.obs["leiden"] + "_" + stage_adata.obs["ident"].astype(str)
        )

        leaf_labels = []
        for leaf in leaves:
            leaf_str = str(leaf)
            sub = stage_adata[stage_adata.obs["leiden"] == leaf_str]
            leaf_labels.append(sub.obs["_leiden_celltype"].iloc[0] if sub.n_obs else leaf_str)

        fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
        sch.dendrogram(Z, ax=ax)
        ax.set_xticklabels(leaf_labels, rotation=90)
        ax.set_title(f"stage={stage}")
        fig.tight_layout()
        path = figure_dir / f"dendrogram_stage_{stage}.pdf"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        _close_fig(fig)
        out_files.append(str(path))

    if not out_files:
        raise ValueError("No dendrogram stages were generated")
    return out_files


def generate_analysis_visualizations(analysis_dir: str | Path) -> dict[str, Any]:
    analysis_dir = Path(analysis_dir).expanduser().resolve()
    figure_dir = analysis_dir / "visualizations"
    tables_dir = figure_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    adata, dataset_path, attribute_path = _read_analysis_adata(analysis_dir)
    prefix = analysis_dir.name

    manifest: dict[str, Any] = {
        "analysis_dir": str(analysis_dir),
        "adata_source": str(dataset_path),
        "attribute_source": str(attribute_path) if attribute_path else None,
        "figure_dir": str(figure_dir),
        "tables_dir": str(tables_dir),
        "successes": [],
        "failures": [],
        "skipped": [],
        "available_uns_keys": list(adata.uns.keys()),
    }

    def record_success(name: str, **payload: Any) -> None:
        entry = {"name": name}
        entry.update(payload)
        manifest["successes"].append(entry)

    def record_failure(name: str, exc: Exception) -> None:
        manifest["failures"].append({"name": name, "error": f"{type(exc).__name__}: {exc}"})

    def record_skip(name: str, reason: str) -> None:
        manifest["skipped"].append({"name": name, "reason": str(reason)})

    try:
        fig, _axes = plot_stage_umap_grid(adata, show=False)
        out = figure_dir / f"{prefix}_stage_umap_grid.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        _close_fig(fig)
        record_success("stage_umap_grid", file=str(out))
    except Exception as exc:
        record_failure("stage_umap_grid", exc)

    try:
        fig, _ax, ctab = plot_stage_cell_type_composition(adata, show=False)
        out = figure_dir / f"{prefix}_stage_composition.png"
        table = tables_dir / f"{prefix}_stage_composition_table.csv"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        _close_fig(fig)
        record_success("stage_composition", file=str(out), table=_write_csv(table, ctab.reset_index()))
    except Exception as exc:
        record_failure("stage_composition", exc)

    for mode, uns_key in (
        ("pathway", "pathway_perturbation_score"),
        ("drug", "drug_perturbation_score"),
        ("single_gene", "single_gene_perturbation_score"),
        ("gene_combo", "two_genes_perturbation_score"),
        ("two_drugs", "two_drugs_perturbation_score"),
    ):
        if uns_key not in adata.uns:
            record_skip(f"{mode}_perturbation", f"{uns_key} is not available in adata.uns")
            continue
        try:
            perturb_change = _resolve_perturb_change(adata, uns_key)
            table_df = extract_perturbation_table(
                adata,
                mode=mode,
                perturb_change=perturb_change,
                track="overall",
            )
            table_path = tables_dir / f"{prefix}_{mode}_perturbation_table.csv"
            record_success(
                f"{mode}_perturbation_table",
                table=_write_csv(table_path, table_df),
                rows=int(len(table_df)),
            )
            use_pval = mode != "pathway"
            fig, _ax, _work = plot_perturbation_ranked_bar(table_df, use_neglog10_pval=use_pval, show=False)
            out = figure_dir / f"{prefix}_{mode.replace('single_gene', 'gene').replace('gene_combo', 'gene_combo')}_ranked_bar.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            _close_fig(fig)
            record_success(f"{mode}_ranked_bar", file=str(out))
        except Exception as exc:
            record_failure(f"{mode}_perturbation", exc)

    if "progressionMarkers" not in adata.uns:
        record_skip("dynamic_markers", "progressionMarkers is not available in adata.uns")
    else:
        try:
            track = _first_progression_track(adata)
            track_token = _filename_token(track, fallback="track")
            dyn_table = extract_dynamic_marker_table(adata, track=track, direction="increasing")
            table_path = tables_dir / f"{prefix}_dynamic_marker_table_track_{track_token}.csv"
            record_success(
                "dynamic_marker_table",
                table=_write_csv(table_path, dyn_table),
                track=track,
                rows=int(len(dyn_table)),
            )

            fig, _axes, dyn_long = plot_dynamic_marker_trajectories(
                adata,
                tracks=[track],
                directions=("increasing",),
                show=False,
            )
            out = figure_dir / f"{prefix}_dynamic_marker_trajectories_track_{track_token}.png"
            long_path = tables_dir / f"{prefix}_dynamic_marker_trajectories_long_track_{track_token}.csv"
            fig.savefig(out, dpi=100, bbox_inches="tight")
            _close_fig(fig)
            record_success(
                "dynamic_marker_trajectories",
                file=str(out),
                table=_write_csv(long_path, dyn_long),
                track=track,
            )

            fig, _axes, _matrix = plot_dynamic_marker_heatmap(
                adata,
                tracks=[track],
                directions=("increasing",),
                show=False,
            )
            out = figure_dir / f"{prefix}_dynamic_marker_heatmap_track_{track_token}.png"
            panels_path = tables_dir / f"{prefix}_dynamic_marker_heatmap_panels_track_{track_token}.csv"
            fig.savefig(out, dpi=100, bbox_inches="tight")
            _close_fig(fig)
            _fig2, _axes2, panels = plot_dynamic_marker_heatmap_track_panels(
                adata,
                tracks=[track],
                directions=("increasing",),
                show=False,
            )
            _close_fig(_fig2)
            record_success(
                "dynamic_marker_heatmap",
                file=str(out),
                table=_write_csv(panels_path, panels),
                track=track,
            )
        except Exception as exc:
            record_failure("dynamic_markers", exc)

    if "hcmarkers" not in adata.uns:
        record_skip("hierarchical_markers", "hcmarkers is not available in adata.uns")
    else:
        try:
            stage, cluster, level = _first_hc_branch(adata)
            marker_table = extract_hierarchical_marker_table(
                adata,
                stage=stage,
                cluster=cluster,
                level=level,
                direction="up",
            )
            fig, _ax, matrix, _markers = plot_hierarchical_marker_branch_heatmap(
                adata,
                stage=stage,
                cluster=cluster,
                level=level,
                direction="up",
                show=False,
            )
            out = figure_dir / f"{prefix}_hierarchical_marker_heatmap_stage{stage}_cluster{cluster}_level{level}.png"
            table_path = tables_dir / f"{prefix}_hierarchical_marker_table_stage{stage}_cluster{cluster}_level{level}.csv"
            matrix_path = tables_dir / f"{prefix}_hierarchical_marker_matrix_stage{stage}_cluster{cluster}_level{level}.csv"
            fig.savefig(out, dpi=100, bbox_inches="tight")
            _close_fig(fig)
            record_success(
                "hierarchical_marker_heatmap",
                file=str(out),
                table=_write_csv(table_path, marker_table),
                matrix=_write_csv(matrix_path, matrix.reset_index()),
                stage=str(stage),
                cluster=str(cluster),
                level=str(level),
            )
        except Exception as exc:
            record_failure("hierarchical_markers", exc)

    if "hcmarkers" not in adata.uns:
        record_skip("hc_dendrograms", "hcmarkers is not available in adata.uns")
    else:
        try:
            files = _generate_hc_dendrograms(adata, figure_dir)
            record_success("hc_dendrograms", files=files)
        except Exception as exc:
            record_failure("hc_dendrograms", exc)

    try:
        fig, _ax, nodes, edges = plot_temporal_dynamics_graph(adata, show=False)
        out = figure_dir / f"{prefix}_temporal_dynamics_graph.png"
        nodes_path = tables_dir / f"{prefix}_temporal_dynamics_all_tracks_nodes.csv"
        edges_path = tables_dir / f"{prefix}_temporal_dynamics_all_tracks_edges.csv"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        _close_fig(fig)
        record_success(
            "temporal_dynamics_graph",
            file=str(out),
            nodes=_write_csv(nodes_path, nodes),
            edges=_write_csv(edges_path, edges),
            scope="all_tracks",
        )
    except Exception as exc:
        record_failure("temporal_dynamics_graph", exc)

    manifest["counts"] = {
        "successes": len(manifest["successes"]),
        "failures": len(manifest["failures"]),
        "skipped": len(manifest["skipped"]),
    }
    manifest_path = figure_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
