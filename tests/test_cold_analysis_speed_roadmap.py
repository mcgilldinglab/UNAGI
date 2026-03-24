import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import UNAGI.UNAGI_analyst as analyst_module
import UNAGI.dynamic_regulatory_networks.processIDREM as process_idrem_module
from UNAGI.marker_discovery.hierachical_static_markers import compare_hcmarker_strategies
from UNAGI.utils.idrem_helper import clear_idrem_json_cache, readIdremJson

from .test_analysis_perturbation_schedule import (
    _FakePerturbationRunner,
    _create_analysis_workspace,
    _patch_analysis_dependencies,
)


def _make_hierarchical_adata() -> AnnData:
    x = sparse.csr_matrix(
        np.array(
            [
                [5.0, 1.0, 0.5, 0.2],
                [4.8, 1.1, 0.4, 0.3],
                [1.0, 4.5, 0.6, 0.3],
                [0.9, 4.7, 0.7, 0.2],
                [4.7, 1.2, 0.4, 0.5],
                [4.9, 1.0, 0.5, 0.4],
                [1.2, 4.8, 0.8, 0.2],
                [1.1, 4.9, 0.7, 0.3],
            ],
            dtype=np.float32,
        )
    )
    adata = AnnData(x)
    adata.obs = pd.DataFrame(
        {
            "stage": ["0", "0", "0", "0", "1", "1", "1", "1"],
            "leiden": pd.Categorical(["0", "0", "1", "1", "0", "0", "1", "1"]),
            "ident": ["A", "A", "B", "B", "A", "A", "B", "B"],
        },
        index=[f"cell{i}" for i in range(8)],
    )
    adata.var = pd.DataFrame(index=["GENE0", "GENE1", "GENE2", "GENE3"])
    adata.obsm["X_umap"] = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [2.0, 0.1],
            [2.1, 0.0],
            [0.0, 2.0],
            [0.1, 2.1],
            [2.0, 2.0],
            [2.1, 2.1],
        ],
        dtype=np.float32,
    )
    return adata


def test_start_analyse_writes_stage_timings_artifact(monkeypatch, tmp_path):
    data_path, target_dir, cmap_dir = _create_analysis_workspace(tmp_path)
    _patch_analysis_dependencies(monkeypatch)
    _FakePerturbationRunner.reset()
    monkeypatch.setattr(
        analyst_module,
        "get_dataset_hcmarkers",
        lambda *args, **kwargs: {"0": {"Z": np.array([[0.0, 1.0, 0.0, 0.0]]), "order": ["0"], "markers": {}}},
    )
    monkeypatch.setattr(
        analyst_module,
        "get_progressionmarker_background",
        lambda *args, **kwargs: {"0-1": [np.zeros(2)]},
    )
    monkeypatch.setattr(
        analyst_module,
        "runGetProgressionMarker_one_dist",
        lambda *args, **kwargs: {"0-1": {"increasing": {}, "decreasing": {}}},
    )

    analyst = analyst_module.analyst(data_path, 1, target_dir=target_dir, cmap_dir=cmap_dir)
    analyst.start_analyse(
        progressionmarker_background_sampling=3,
        run_perturbation=True,
        random_times=3,
        ignore_dynamic_markers=False,
        ignore_hcmarkers=False,
        perturbation_schedule="single_gpu_optimized",
        dynamic_marker_strategy="optimized",
        hierarchical_marker_cpu_workers=1,
    )

    timings_path = target_dir / "analysis_stage_timings.json"
    assert timings_path.exists()
    payload = json.loads(timings_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["resolved_execution_backend"] in {"gpu_exact", "cpu_exact_parallel", "cpu_centroid", "gpu_centroid"}
    assert "hierarchical_markers" in payload["stages"]
    assert "dynamic_markers_background" in payload["stages"]
    assert "dynamic_markers_scoring" in payload["stages"]
    assert "pathway_target_prep" in payload["stages"]
    assert "pathway_background" in payload["stages"]
    assert "pathway_scoring" in payload["stages"]
    assert "drug_target_prep" in payload["stages"]
    assert "drug_background" in payload["stages"]
    assert "drug_scoring" in payload["stages"]
    assert "save_merge" in payload["stages"]


def test_hierarchical_optimized_matches_legacy_in_serial_and_parallel():
    adata = _make_hierarchical_adata()
    comparison = compare_hcmarker_strategies(
        adata,
        stage_key="stage",
        cluster_key="leiden",
        use_rep="umaps",
        cutoff=1.0,
        cpu_workers=2,
    )
    assert comparison["ok"] is True
    assert comparison["comparisons"]["legacy_vs_optimized_serial"]["ok"] is True
    assert comparison["comparisons"]["legacy_vs_optimized_parallel"]["ok"] is True


def test_idrem_json_cache_invalidates_when_drem_json_changes(tmp_path):
    track_dir = tmp_path / "idrem" / "0-0-0.txt_viz"
    track_dir.mkdir(parents=True, exist_ok=True)
    drem_path = track_dir / "DREM.json"
    clear_idrem_json_cache()
    payload1 = [[{"nodeID": 1, "nodeMean": 0.1}], None, [], ["GENE0"], [], [], None, None, [["gene", "stage0"], ["GENE0", "0.1"]]]
    drem_path.write_text(f"data={json.dumps(payload1)};", encoding="utf-8")
    parsed1 = readIdremJson(track_dir.parent, "0-0-0.txt_viz")
    parsed2 = readIdremJson(track_dir.parent, "0-0-0.txt_viz")
    assert parsed1 is parsed2

    time.sleep(0.01)
    payload2 = [[{"nodeID": 1, "nodeMean": 0.9}], None, [], ["GENE0"], [], [], None, None, [["gene", "stage0"], ["GENE0", "0.9"]]]
    drem_path.write_text(f"data={json.dumps(payload2)};", encoding="utf-8")
    parsed3 = readIdremJson(track_dir.parent, "0-0-0.txt_viz")
    assert parsed3[0][0]["nodeMean"] == 0.9


def test_run_idrem_writes_timing_manifest(monkeypatch, tmp_path):
    idrem_dir = tmp_path / "idrem_assets"
    idrem_dir.mkdir(parents=True, exist_ok=True)
    (idrem_dir / "example_settings.txt").write_text("\n".join(["line"] * 60), encoding="utf-8")
    midpath = tmp_path / "mid"
    iteration = 1

    def _fake_thread_run(self):
        stem = Path(self.indir).stem
        viz_dir = Path(self.indir).parent.parent / "idremInput" / f"{stem}.txt_viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        (viz_dir / "DREM.json").write_text("data=[];", encoding="utf-8")

    monkeypatch.setattr(process_idrem_module.IDREMthread, "run", _fake_thread_run)
    monkeypatch.setattr(process_idrem_module, "test_idrem_results", lambda path, filename: True)

    process_idrem_module.runIdrem(
        paths=[[[0], [0]]],
        midpath=midpath,
        idremInput=[np.array([[1.0, 2.0]])],
        genenames=["GENE0"],
        iteration=iteration,
        idrem_dir=idrem_dir,
        species="Human",
    )

    timing_path = midpath / str(iteration) / "idrem_timing.json"
    assert timing_path.exists()
    payload = json.loads(timing_path.read_text(encoding="utf-8"))
    assert "settings_generation_seconds" in payload
    assert "external_run_seconds" in payload
    assert "result_collection_seconds" in payload
    assert "validation_seconds" in payload
