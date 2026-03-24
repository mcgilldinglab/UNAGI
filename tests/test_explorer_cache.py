from __future__ import annotations

import os
from pathlib import Path

import pytest

from UNAGI.monitoring.explorer_cache import UmapViewportCache

from .interactive_analysis_test_helpers import write_analysis_dataset


def test_umap_cache_builds_and_invalidates_manifest_stage_shards_and_gene_vectors(tmp_path):
    skill_root = tmp_path / "skill_root"
    analysis_dir = write_analysis_dataset(tmp_path / "analysis" / "iter1", cells_per_stage=48)
    cache = UmapViewportCache(skill_root=skill_root)

    manifest = cache.get_manifest(analysis_dir)
    stage_shard = cache.get_stage_shard(analysis_dir, stage="0")
    gene_values = cache.get_gene_vector(analysis_dir, stage="0", gene="gene0")

    cache_dir = skill_root / "outputs" / "explorer_cache" / manifest["analysis_hash"]
    assert (cache_dir / "manifest.json").exists()
    stage_dir = cache_dir / "stages" / manifest["stage_info"]["0"]["slug"]
    assert (stage_dir / "stage_shard.npz").exists()
    assert (stage_dir / "stage_meta.json").exists()
    gene_dir = cache_dir / "genes" / manifest["stage_info"]["0"]["slug"]
    assert any(gene_dir.glob("*.npy"))
    assert len(stage_shard.cell_ids) == 48
    assert gene_values.shape == (48,)

    attribute_path = Path(manifest["attribute_path"])
    stat = attribute_path.stat()
    os.utime(attribute_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    refreshed = cache.get_manifest(analysis_dir)
    assert refreshed["analysis_hash"] != manifest["analysis_hash"]


def test_umap_cache_switches_between_density_and_points_and_summarizes_selection(tmp_path):
    skill_root = tmp_path / "skill_root"
    analysis_dir = write_analysis_dataset(
        tmp_path / "analysis" / "iter1",
        stage_labels=("0",),
        cells_per_stage=60_000,
    )
    cache = UmapViewportCache(skill_root=skill_root)

    density_payload = cache.get_payload(analysis_dir, stage="0", color_by="ident")
    assert density_payload["render_mode"] == "sampled_points"
    assert density_payload["points"]
    assert density_payload["density"] is None
    assert density_payload["selection_supported"] is False

    gene_density = cache.get_payload(analysis_dir, stage="0", color_by="ident", gene="gene0")
    assert gene_density["render_mode"] == "density"
    assert gene_density["expression_gene"] == "GENE0"
    assert gene_density["density"]["metric"] == "mean_expression"

    stage_shard = cache.get_stage_shard(analysis_dir, stage="0")
    x = float(stage_shard.coords[0, 0])
    y = float(stage_shard.coords[0, 1])
    points_payload = cache.get_payload(
        analysis_dir,
        stage="0",
        color_by="ident",
        x_min=x - 1e-6,
        x_max=x + 1e-6,
        y_min=y - 1e-6,
        y_max=y + 1e-6,
        current_render_mode="sampled_points",
    )
    assert points_payload["render_mode"] == "points"
    assert points_payload["point_count"] >= 1

    selected_cell_id = points_payload["points"][0]["cell_id"]
    metadata_summary = cache.summarize_selection(
        analysis_dir,
        stage="0",
        color_by="ident",
        cell_ids=[selected_cell_id],
        render_mode="points",
    )
    assert metadata_summary["summary_type"] == "metadata"
    assert metadata_summary["selected_count"] == 1
    assert metadata_summary["rows"]

    gene_summary = cache.summarize_selection(
        analysis_dir,
        stage="0",
        color_by="ident",
        cell_ids=[selected_cell_id],
        gene="gene0",
        render_mode="points",
    )
    assert gene_summary["summary_type"] == "gene"
    assert gene_summary["rows"][0][0] == "GENE0"

    with pytest.raises(ValueError, match="point mode"):
        cache.summarize_selection(
            analysis_dir,
            stage="0",
            color_by="ident",
            cell_ids=[selected_cell_id],
            render_mode="density",
        )
