from __future__ import annotations

import importlib
import html
import json
import re
import sys
from pathlib import Path

import anyio
import httpx
import pandas as pd
import pytest

import UNAGI.monitoring.idrem_summary as idrem_summary_module
import UNAGI.monitoring.interactive_analysis as interactive_analysis_module
import UNAGI.monitoring.webapp as webapp_module
from UNAGI.monitoring.explorer_cache import UmapViewportCache
import UNAGI.monitoring.service as service_module
from UNAGI.monitoring.webapp import create_monitor_app

from .interactive_analysis_test_helpers import write_analysis_dataset
from .monitor_test_helpers import (
    SKILL_DIR,
    create_analysis_job,
    create_custom_perturbation_job,
    create_monitor_workspace,
    create_train_job,
    fake_toppgene_enrichment_table,
    write_mock_idrem_track,
)

SCRIPTS_DIR = SKILL_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

job_backend = importlib.import_module("unagi_job")


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _patch_monitor_telemetry(monkeypatch):
    monkeypatch.setattr(
        service_module,
        "collect_system_metrics",
        lambda skill_root: {
            "cpu_percent": 22.0,
            "memory": {"percent": 48.0},
            "disk": {"percent": 63.0},
            "gpus": [
                {
                    "index": 0,
                    "name": "NVIDIA GeForce RTX 4090",
                    "utilization_gpu_percent": 55.0,
                    "memory_used_mb": 1024.0,
                    "memory_total_mb": 24576.0,
                    "temperature_c": 60.0,
                }
            ],
            "gpu_error": None,
            "gpu_pid_map": {4321: {"gpu_indices": [0], "memory_used_mb": 512.0}},
        },
    )
    monkeypatch.setattr(
        service_module,
        "collect_process_metrics",
        lambda pid, gpu_pid_map=None: (
            {
                "pid": int(pid),
                "cpu_percent": 9.5,
                "memory_rss_bytes": 8192,
                "status": "running",
                "gpu": {"gpu_indices": [0], "memory_used_mb": 512.0},
            }
            if pid
            else None
        ),
    )


@pytest.mark.anyio
async def test_monitor_web_routes_render_dashboard_detail_partials_and_api(tmp_path, monkeypatch):
    _patch_monitor_telemetry(monkeypatch)
    skill_root, repo_root, _ = create_monitor_workspace(tmp_path)
    train = create_train_job(
        skill_root,
        job_id="train_001",
        job_name="Dashboard Training Job",
        status="running",
        log_text=(
            "[UNAGI-TRAIN] iteration=0 max_iter=2 stage=load_staged_data status=START\n"
            "[UNAGI-TRAIN] iteration=0 max_iter=2 stage=load_staged_data status=DONE\n"
            "[UNAGI-TRAIN] iteration=0 max_iter=2 stage=training_model status=START\n"
            "[UNAGI-TRAIN] iteration=0 epoch=2/5 loss=123.4\n"
        ),
    )
    analysis = create_analysis_job(
        train["job_dir"],
        job_id="analyze_001",
        job_name="Dashboard Analysis Job",
        parent_job_id="train_001",
        iteration=0,
        status="succeeded",
        log_text=(
            "[UNAGI-ANALYSIS] start analyse iteration=0, run_perturbation=False, centroid=False, "
            "perturb_change=0.5, direction_mode=two_sided\n"
            "[ANALYSIS] DONE\n"
        ),
    )
    create_custom_perturbation_job(
        analysis["job_dir"],
        analysis["result_dir"],
        job_id="perturb_001",
        job_name="Dashboard Perturbation Job",
        parent_job_id="analyze_001",
        iteration=0,
        status="succeeded",
        log_text="[UNAGI-CUSTOM-PERTURB] stage=save status=DONE\n",
    )

    app = create_monitor_app(skill_root=skill_root, repo_root=repo_root, skill_dir=SKILL_DIR)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/")
        assert response.status_code == 200
        assert "Durable Job Dashboard" in response.text
        assert "Dashboard Training Job" in response.text

        response = await client.get("/jobs/train_001")
        assert response.status_code == 200
        assert "Dashboard Training Job" in response.text
        assert "Dashboard Analysis Job" in response.text
        assert "/browse?path=" in response.text

        response = await client.get("/jobs/analyze_001")
        assert response.status_code == 200
        assert "Dashboard Perturbation Job" in response.text

        response = await client.get("/partials/dashboard")
        assert response.status_code == 200
        assert "Training Runs" in response.text
        assert "Analysis Jobs" in response.text
        assert "Dashboard Analysis Job" in response.text

        response = await client.get("/partials/system")
        assert response.status_code == 200
        assert "NVIDIA GeForce RTX 4090" in response.text
        assert "Tracked Job Processes" in response.text

        response = await client.get("/partials/jobs/train_001/summary")
        assert response.status_code == 200
        assert "Active Process" in response.text
        assert "Key Paths" not in response.text

        response = await client.get("/partials/jobs/train_001/log")
        assert response.status_code == 200
        assert "[UNAGI-TRAIN] iteration=0 epoch=2/5 loss=123.4" in response.text

        response = await client.get("/api/jobs")
        assert response.status_code == 200
        jobs = response.json()
        assert {job["job_id"] for job in jobs} == {"train_001", "analyze_001", "perturb_001"}

        response = await client.get("/api/jobs/train_001")
        assert response.status_code == 200
        job = response.json()
        assert job["job_id"] == "train_001"
        assert job["process"]["gpu"]["gpu_indices"] == [0]

        response = await client.get("/api/system")
        assert response.status_code == 200
        assert response.json()["gpus"][0]["name"] == "NVIDIA GeForce RTX 4090"


@pytest.mark.anyio
async def test_monitor_browse_route_reads_files_and_directories(tmp_path, monkeypatch):
    _patch_monitor_telemetry(monkeypatch)
    skill_root, repo_root, _ = create_monitor_workspace(tmp_path)
    train = create_train_job(
        skill_root,
        job_id="train_002",
        job_name="Browse Job",
        status="succeeded",
        finished_at="2026-03-06T10:30:00+00:00",
        log_text="[ANALYSIS] DONE\n",
    )

    app = create_monitor_app(skill_root=skill_root, repo_root=repo_root, skill_dir=SKILL_DIR)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/browse", params={"path": str(train["config_path"])})
        assert response.status_code == 200
        assert "epoch_initial" in response.text

        response = await client.get("/browse", params={"path": str(train["output_path"])})
        assert response.status_code == 200
        assert "model_save" in response.text


@pytest.mark.anyio
async def test_monitor_browse_route_allows_scanned_job_roots(tmp_path, monkeypatch):
    _patch_monitor_telemetry(monkeypatch)
    skill_root, repo_root, _ = create_monitor_workspace(tmp_path)
    external_root = tmp_path / "external_workspace"
    external_train = create_train_job(
        external_root,
        job_id="train_external",
        job_name="External Browse Job",
        status="succeeded",
        finished_at="2026-03-06T10:30:00+00:00",
        log_text="[ANALYSIS] DONE\n",
    )
    external_jobs_root = external_root / "outputs" / "skill_jobs"

    app = create_monitor_app(
        skill_root=skill_root,
        repo_root=repo_root,
        skill_dir=SKILL_DIR,
        job_roots=[external_jobs_root],
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/browse", params={"path": str(external_train["output_path"] / "model_save")})
        assert response.status_code == 200


@pytest.mark.anyio
async def test_custom_perturbation_submit_works_for_scanned_analysis_job_root(tmp_path, monkeypatch):
    _patch_monitor_telemetry(monkeypatch)
    skill_root, repo_root, _ = create_monitor_workspace(tmp_path)
    external_root = tmp_path / "external_workspace"
    external_train = create_train_job(
        external_root,
        job_id="train_external",
        job_name="External Training Job",
        status="succeeded",
        finished_iterations=[1],
        log_text="Iteration 1 training is done!\n",
    )
    external_analysis = create_analysis_job(
        external_train["job_dir"],
        job_id="analyze_external",
        job_name="External Analysis Job",
        parent_job_id="train_external",
        iteration=1,
        status="succeeded",
        stale_paths=True,
        log_text="[ANALYSIS] DONE\n",
    )
    write_analysis_dataset(external_analysis["result_dir"])
    monkeypatch.setattr(
        job_backend,
        "_ensure_custom_perturbation_daemon",
        lambda **kwargs: (5252, ["python", "perturb-daemon", "--queue-key", kwargs["queue_key"]]),
    )

    app = create_monitor_app(
        skill_root=skill_root,
        repo_root=repo_root,
        skill_dir=SKILL_DIR,
        job_roots=[external_root / "outputs" / "skill_jobs"],
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/analysis/analyze_external/perturbations",
            data={
                "mode": "gene",
                "scope_kind": "overall",
                "perturb_change": "0.5",
                "direction_mode": "two_sided",
                "execution_mode": "auto",
                "signed_gene_text": "gene0:+",
            },
        )
        assert response.status_code == 201
        payload = response.json()
        assert payload["status"] == "submitted"

    perturb_job_dir = external_analysis["job_dir"] / "perturbation_jobs" / payload["job_id"]
    assert perturb_job_dir.exists()


@pytest.mark.anyio
async def test_analysis_share_bundle_export_route_and_page(tmp_path, monkeypatch):
    _patch_monitor_telemetry(monkeypatch)
    skill_root, repo_root, _ = create_monitor_workspace(tmp_path)
    train = create_train_job(
        skill_root,
        job_id="train_share_bundle",
        job_name="Share Bundle Training Job",
        status="succeeded",
        finished_iterations=[1],
        log_text="Iteration 1 training is done!\n",
    )
    analysis = create_analysis_job(
        train["job_dir"],
        job_id="analyze_share_bundle",
        job_name="Share Bundle Analysis Job",
        parent_job_id="train_share_bundle",
        iteration=1,
        status="succeeded",
        with_visualization_manifest=True,
        log_text="[ANALYSIS] DONE\n",
    )
    write_analysis_dataset(analysis["result_dir"], include_stage_specific_perturbation=True)
    write_mock_idrem_track(analysis["result_dir"], track_key="0-2-4")

    def fake_export(analysis_dir, *, force=False, analysis_label=None):
        analysis_path = Path(analysis_dir)
        bundle_dir = analysis_path / "share_bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / "index.html").write_text("<html><body>bundle</body></html>", encoding="utf-8")
        zip_path = analysis_path / "unagi_offline_report_Share_Bundle_Analysis_Job_20260324T120000Z.zip"
        zip_path.write_text("zip", encoding="utf-8")
        manifest = {
            "analysis_dir": str(analysis_path),
            "analysis_label": analysis_label or analysis_path.name,
            "bundle_dir": str(bundle_dir),
            "index_path": str(bundle_dir / "index.html"),
            "zip_path": str(zip_path),
            "zip_name": zip_path.name,
            "counts": {"entries": 3, "tracks": 1},
        }
        (bundle_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    monkeypatch.setattr(webapp_module, "export_analysis_share_bundle", fake_export)

    app = create_monitor_app(skill_root=skill_root, repo_root=repo_root, skill_dir=SKILL_DIR)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/api/analysis/analyze_share_bundle/share-bundle/export")
        assert response.status_code == 202
        payload = response.json()
        assert payload["status"] in {"queued", "running", "succeeded"}
        task_id = payload["task_id"]

        final = None
        for _ in range(20):
            poll = await client.get(f"/api/direct-tasks/{task_id}")
            assert poll.status_code == 200
            final = poll.json()
            if final["status"] == "succeeded":
                break
            await anyio.sleep(0.01)
        assert final is not None
        assert final["status"] == "succeeded"
        assert final["result"]["counts"]["tracks"] == 1

        response = await client.get("/analysis/analyze_share_bundle/visualizations")
        assert response.status_code == 200
        assert "Export Offline Report" in response.text or "Regenerate Offline Report" in response.text
        assert "Download unagi_offline_report_Share_Bundle_Analysis_Job_20260324T120000Z.zip" in response.text


def test_custom_perturbation_queue_diagnostics_for_stuck_job(tmp_path, monkeypatch):
    _patch_monitor_telemetry(monkeypatch)
    skill_root, repo_root, _ = create_monitor_workspace(tmp_path)
    train = create_train_job(
        skill_root,
        job_id="train_queue_diag",
        job_name="Queue Diag Training Job",
        status="succeeded",
        finished_iterations=[1],
        log_text="Iteration 1 training is done!\n",
    )
    analysis = create_analysis_job(
        train["job_dir"],
        job_id="analyze_queue_diag",
        job_name="Queue Diag Analysis Job",
        parent_job_id="train_queue_diag",
        iteration=1,
        status="succeeded",
        log_text="[ANALYSIS] DONE\n",
    )
    write_analysis_dataset(analysis["result_dir"], include_stage_specific_perturbation=True)
    create_custom_perturbation_job(
        analysis["job_dir"],
        analysis["result_dir"],
        job_id="perturb_stuck",
        job_name="TGFBETA",
        parent_job_id="analyze_queue_diag",
        iteration=1,
        status="queued",
        finished_at=None,
        log_text="",
    )

    diagnostics = job_backend.custom_perturbation_queue_diagnostics(
        repo_root=repo_root,
        job_file_path=analysis["job_dir"] / "perturbation_jobs" / "perturb_stuck" / "job.json",
    )
    assert diagnostics["derived_status"] == "stuck"
    assert diagnostics["message"] in {
        "Queued, but no active worker is watching this run root.",
        "Queued, worker not running.",
    }


@pytest.mark.anyio
async def test_analysis_visualization_page_includes_interactive_explorer_and_render_routes(tmp_path, monkeypatch):
    _patch_monitor_telemetry(monkeypatch)
    monkeypatch.setattr(
        idrem_summary_module.ToppGeneEnrichmentClient,
        "fetch",
        lambda self, genes: fake_toppgene_enrichment_table(),
    )
    skill_root, repo_root, _ = create_monitor_workspace(tmp_path)
    train = create_train_job(
        skill_root,
        job_id="train_003",
        job_name="Explorer Training Job",
        status="succeeded",
        finished_at="2026-03-06T10:30:00+00:00",
        log_text="Iteration 1 training is done!\n",
    )
    analysis = create_analysis_job(
        train["job_dir"],
        job_id="analyze_explorer",
        job_name="Explorer Analysis Job",
        parent_job_id="train_003",
        iteration=1,
        status="succeeded",
        with_visualization_manifest=True,
        log_text="[ANALYSIS] DONE\n",
    )
    write_analysis_dataset(analysis["result_dir"], include_stage_specific_perturbation=True)
    write_mock_idrem_track(analysis["result_dir"], track_key="0-2-4")
    track_table_path = analysis["result_dir"] / "stage_specific_perturbation" / "stage_1" / "pathway_perturbation_track_group_0_stage_1.csv"

    def fake_generate_stage_track_table(analysis_dir_arg, *, mode, stage, track_group):
        frame = pd.DataFrame(
            {
                "rank": [0, 1],
                "pathways": ["PathTrackA", "PathTrackB"],
                "perturbation score": [1.8, 1.1],
                "pval_adjusted": [0.003, 0.02],
                "regulated genes": ["['GENE0']", "['GENE1']"],
                "idrem_suggestion": ["['GENE0:+']", "['GENE1:-']"],
            }
        )
        track_table_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(track_table_path, index=False)
        return str(track_table_path)

    monkeypatch.setattr(
        interactive_analysis_module,
        "_ensure_stage_specific_track_group_perturbation_table",
        fake_generate_stage_track_table,
    )
    create_custom_perturbation_job(
        analysis["job_dir"],
        analysis["result_dir"],
        job_id="perturb_done",
        job_name="Online Gene Perturbation",
        parent_job_id="analyze_explorer",
        iteration=1,
        status="succeeded",
        log_text="[UNAGI-CUSTOM-PERTURB] stage=save status=DONE\n",
    )
    visualizations_dir = analysis["result_dir"] / "visualizations"
    tables_dir = visualizations_dir / "tables"
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    graph_png = visualizations_dir / "results_temporal_dynamics_graph.png"
    stage_png = visualizations_dir / "results_stage_composition.png"
    graph_png.write_text("stub", encoding="utf-8")
    stage_png.write_text("stub", encoding="utf-8")
    (visualizations_dir / "manifest.json").write_text(
        json.dumps(
            {
                "analysis_dir": str(analysis["result_dir"]),
                "figure_dir": str(visualizations_dir),
                "tables_dir": str(tables_dir),
                "successes": [
                    {"name": "temporal_dynamics_graph", "file": str(graph_png)},
                    {"name": "stage_composition", "file": str(stage_png)},
                ],
                "failures": [],
                "counts": {"successes": 2, "failures": 0},
            }
        ),
        encoding="utf-8",
    )

    app = create_monitor_app(skill_root=skill_root, repo_root=repo_root, skill_dir=SKILL_DIR)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/analysis/analyze_explorer/visualizations")
        assert response.status_code == 200
        assert "Interactive Explorer" in response.text
        assert "Loading interactive temporal dynamics graph" in response.text
        assert "Loading interactive stage composition" in response.text
        assert 'id="temporal-overview-tabs-analyze_explorer"' in response.text
        assert 'data-tab-target="temporal-graph-panel-analyze_explorer"' in response.text
        assert 'data-tab-target="stage-composition-panel-analyze_explorer"' in response.text
        assert "Hover a node to inspect stage, cluster, and ident." not in response.text
        assert "Selected Node" not in response.text
        assert "/api/analysis/analyze_explorer/temporal-graph" in response.text
        assert "/api/analysis/analyze_explorer/stage-composition" in response.text
        assert '/analysis/analyze_explorer/idrem/track-popup' in response.text
        assert "Click a temporal node with a track to open its iDREM summary in a popup window." not in response.text
        assert "/partials/analysis/analyze_explorer/explorer" in response.text

        response = await client.get(
            "/analysis/analyze_explorer/idrem/track-popup",
            params={"track_key": "0-2-4", "track_label": "track 0 AT1"},
        )
        assert response.status_code == 200
        assert "Track iDREM Summary" in response.text
        assert 'data-idrem-track-panel' in response.text
        assert '/api/analysis/analyze_explorer/idrem/track?track_key=0-2-4&amp;track_label=track+0+AT1' in response.text

        response = await client.get("/partials/analysis/analyze_explorer/explorer")
        assert response.status_code == 200
        explorer_html = response.text
        assert "explorer-shell" in response.text
        assert "explorer-sidebar" in response.text
        assert "explorer-content" in response.text
        assert "Dynamic Markers" in response.text
        assert "HIERARCHICAL MARKERS" in response.text
        assert "Enable UMAP Tab" in response.text
        assert 'name="umap_stage"' not in response.text

        response = await client.get("/partials/analysis/analyze_explorer/explorer", params={"mode": "umap", "enable_umap": "true"})
        assert response.status_code == 200
        assert 'name="enable_umap" value="true"' in response.text
        assert 'name="umap_stage"' in response.text
        assert 'data-umap-color-mode-option="metadata"' in response.text
        assert 'data-umap-color-mode-option="gene"' in response.text
        assert "Metadata Field" in response.text
        assert "Gene" in response.text
        assert "data-umap-render-mode" not in response.text
        assert "data-umap-cells-table" not in response.text
        assert "data-umap-gene-clear" in response.text
        assert "data-umap-field-group hidden" not in response.text
        assert "data-umap-gene-group hidden" in response.text
        assert re.search(r"data-umap-gene-clear[\s\S]*?\bhidden\b", response.text)

        response = await client.get("/api/analysis/analyze_explorer/umap")
        assert response.status_code == 200
        payload = response.json()
        assert payload["point_count"] == 24
        assert payload["stage"] == "0"
        assert payload["stage_options"] == ["0", "1", "2"]
        assert "stage" in payload["color_options"]

        response = await client.get("/api/analysis/analyze_explorer/umap", params={"stage": "1"})
        assert response.status_code == 200
        stage_payload = response.json()
        assert stage_payload["point_count"] == 24
        assert stage_payload["stage"] == "1"

        response = await client.get("/api/analysis/analyze_explorer/umap", params={"stage": "1", "gene": "GENE0"})
        assert response.status_code == 200
        gene_payload = response.json()
        assert gene_payload["expression_gene"] == "GENE0"
        assert "expression" in gene_payload["points"][0]

        response = await client.get("/api/analysis/analyze_explorer/temporal-graph")
        assert response.status_code == 200
        temporal_payload = response.json()
        assert temporal_payload["node_count"] >= 2
        assert temporal_payload["edge_count"] >= 1
        assert temporal_payload["nodes"][0]["cell_type"]
        assert temporal_payload["stage_tick_labels"]

        response = await client.get("/api/analysis/analyze_explorer/stage-composition")
        assert response.status_code == 200
        stage_composition_payload = response.json()
        assert stage_composition_payload["data"][0]["type"] == "bar"
        assert stage_composition_payload["layout"]["barmode"] == "stack"

        response = await client.get(
            "/api/analysis/analyze_explorer/hierarchical/dendrogram",
            params={"stage": "0", "cluster": "0"},
        )
        assert response.status_code == 200
        dendrogram_payload = response.json()
        assert dendrogram_payload["data"][0]["type"] == "scatter"
        assert "Hierarchical dendrogram" in dendrogram_payload["layout"]["title"]["text"]

        response = await client.post(
            "/partials/analysis/analyze_explorer/explorer/render",
            data={"mode": "hierarchical", "stage": "0", "cluster": "0", "level": "0", "top_n": "2"},
        )
        assert response.status_code == 200
        heatmap_url_match = re.search(r'data-heatmap-url="([^"]+)"', response.text)
        assert heatmap_url_match
        assert "/api/analysis/analyze_explorer/hierarchical/heatmap?" in html.unescape(heatmap_url_match.group(1))

        response = await client.get(
            "/api/analysis/analyze_explorer/idrem/track",
            params={"track_key": "0-2-4", "track_label": "track 0 AT1"},
        )
        assert response.status_code == 200
        idrem_payload = response.json()
        assert idrem_payload["track_label"] == "track 0 AT1"
        assert idrem_payload["path_labels"] == ["Path A", "Path B"]
        assert idrem_payload["original_html_url"].startswith("/files?path=")
        assert idrem_payload["paths"][0]["enrichment"]["columns"][0] == "Category"
        assert idrem_payload["paths"][0]["enrichment"]["rows"][0][5] == "GENE0"
        assert idrem_payload["paths"][0]["enrichment"]["hover_texts"][0] == "Matched genes: GENE0"
        assert "Top TFs:" in idrem_payload["path_plot"]["paths"][0]["hover_texts"][1]

        response = await client.post(
            "/partials/analysis/analyze_explorer/explorer/render",
            data={
                "mode": "umap",
                "enable_umap": "true",
                "umap_gene": "gene0",
            },
        )
        assert response.status_code == 200
        assert "data-umap-field-group hidden" in response.text
        assert "data-umap-gene-group hidden" not in response.text
        assert 'value="GENE0"' in response.text
        assert not re.search(r"data-umap-gene-clear[\s\S]*?\bhidden\b", response.text)

        response = await client.post(
            "/partials/analysis/analyze_explorer/explorer/render",
            data={
                "mode": "dynamic_markers",
                "dyn_track": "0-2-4",
                "dyn_directions": ["increasing"],
                "dyn_plot_kind": "trajectories",
            },
        )
        assert response.status_code == 200
        assert 'name="dyn_view_mode"' in response.text
        assert "track 0 AT1" in response.text
        assert "/analysis/analyze_explorer/explorer/table.csv?" in response.text
        assert "Underlying data" not in response.text

        response = await client.post(
            "/partials/analysis/analyze_explorer/explorer/render",
            data={
                "mode": "dynamic_markers",
                "dyn_view_mode": "custom_genes",
                "gene_scope": "global_stage",
                "gene_list": "GENE0, GENE1",
            },
        )
        assert response.status_code == 200
        assert "Dynamic Marker Explorer" in response.text
        assert "Trajectory plot" in response.text
        assert "global stage" in response.text
        assert 'name="gene_track"' not in response.text
        assert "temporal track" in response.text
        assert "/analysis/analyze_explorer/explorer/table.csv?" in response.text
        assert "Underlying data" not in response.text

        response = await client.post(
            "/partials/analysis/analyze_explorer/explorer/render",
            data={
                "mode": "dynamic_markers",
                "dyn_view_mode": "custom_genes",
                "gene_scope": "progression_track",
                "gene_track": "0-2-4",
                "gene_list": "GENE0, GENE1",
            },
        )
        assert response.status_code == 200
        assert "temporal track" in response.text
        assert 'name="gene_track"' in response.text

        response = await client.get(
            "/analysis/analyze_explorer/explorer/table.csv",
            params={
                "mode": "dynamic_markers",
                "dyn_view_mode": "ranked_markers",
                "dyn_track": "0-2-4",
                "dyn_directions": ["increasing"],
                "dyn_plot_kind": "trajectories",
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        assert "gene" in response.text.splitlines()[0]

        response = await client.post(
            "/partials/analysis/analyze_explorer/explorer/render",
            data={
                "mode": "hierarchical",
                "hier_stage": "0",
                "hier_cluster": "0",
                "hier_level": "0",
                "hier_top_n": "2",
            },
        )
        assert response.status_code == 200
        assert 'name="hier_direction"' not in response.text
        assert "direction=Up Regulated" not in response.text
        assert "data-hierarchical-shell" in response.text
        assert "/analysis/analyze_explorer/explorer/table.csv?" in response.text
        assert "Selected stage dendrogram" in response.text
        assert "Branch heatmap" in response.text

        response = await client.get(
            "/analysis/analyze_explorer/explorer/table.csv",
            params={
                "mode": "hierarchical",
                "hier_stage": "0",
                "hier_cluster": "0",
                "hier_level": "0",
                "hier_direction": "up",
                "hier_top_n": "2",
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        assert response.text.splitlines()[0].split(",")[0] == "Gene"

        response = await client.post(
            "/partials/analysis/analyze_explorer/explorer/render",
            data={
                "mode": "perturbation",
                "pert_mode": "pathway",
                "perturb_change": "0.5",
                "pert_track": "overall",
                "pert_top_n": "2",
            },
        )
        assert response.status_code == 200
        assert "Perturbation Explorer" in response.text
        assert 'name="pert_panel_mode"' in response.text
        assert 'name="pert_scope"' in response.text
        assert "Custom Perturbation" not in response.text
        assert 'data-custom-perturbation-form' not in response.text
        assert 'data-plotly-widget' in response.text
        assert "perturbation score" in response.text

        response = await client.post(
            "/partials/analysis/analyze_explorer/explorer/render",
            data={
                "mode": "perturbation",
                "pert_scope": "stage",
                "pert_mode": "pathway",
                "pert_stage": "2",
                "pert_top_n": "2",
            },
        )
        assert response.status_code == 200
        assert 'name="pert_stage"' in response.text
        assert ">Stage 0<" not in response.text
        assert "individual stage" in response.text
        assert "scope=stage" in response.text
        assert "Stage 2" in response.text
        assert "/analysis/analyze_explorer/explorer/table.csv?" in response.text

        response = await client.post(
            "/partials/analysis/analyze_explorer/explorer/render",
            data={
                "mode": "perturbation",
                "pert_scope": "stage",
                "pert_mode": "pathway",
                "pert_stage": "1",
                "pert_track": "0",
                "pert_top_n": "2",
            },
        )
        assert response.status_code == 200
        assert 'name="pert_track"' in response.text
        assert "track 0 AT1" in response.text
        assert "scope=stage" in response.text

        response = await client.get(
            "/analysis/analyze_explorer/explorer/table.csv",
            params={
                "mode": "perturbation",
                "pert_scope": "stage",
                "pert_mode": "drug",
                "pert_stage": "1",
                "pert_top_n": "2",
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        assert "DrugStage1A" in response.text

        response = await client.get(
            "/analysis/analyze_explorer/explorer/table.csv",
            params={
                "mode": "perturbation",
                "pert_scope": "stage",
                "pert_mode": "pathway",
                "pert_stage": "1",
                "pert_track": "0",
                "pert_top_n": "2",
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        assert "PathTrackA" in response.text

        response = await client.post(
            "/partials/analysis/analyze_explorer/explorer/render",
            data={
                "mode": "perturbation",
                "pert_panel_mode": "custom",
            },
        )
        assert response.status_code == 200
        assert "Custom Perturbation" in response.text
        assert "Submitted Custom Perturbations" in response.text
        assert 'data-custom-perturbation-form' in response.text
        assert "/api/analysis/analyze_explorer/perturbations/validate" in response.text
        assert "/api/analysis/analyze_explorer/perturbations" in response.text
        assert "/api/analysis/analyze_explorer/perturbations/suggest" in response.text
        assert "successMessage: 'Passed'" in response.text
        assert "refreshExplorerOnSuccess" in response.text
        assert "Delete Selected" in response.text
        assert "different random backgrounds" in response.text
        assert 'data-custom-perturbation-target-input' in response.text
        assert 'data-custom-perturbation-gene-text' in response.text
        assert 'data-custom-perturbation-target-label' in response.text
        assert 'name="stage_key"' in response.text
        assert '>Stage<' in response.text
        assert 'name="job_name"' not in response.text
        assert 'name="user_label"' not in response.text
        assert "Selected Custom Perturbation" not in response.text

        response = await client.get(
            "/partials/analysis/analyze_explorer/explorer",
            params={
                "mode": "perturbation",
                "pert_panel_mode": "custom",
                "selected_custom_job_id": "perturb_done",
            },
        )
        assert response.status_code == 200
        assert "Custom Perturbation Result" in response.text
        assert "Perturbation Scores vs Random Background" in response.text
        assert "Open Job Detail" not in response.text

        running = create_custom_perturbation_job(
            analysis["job_dir"],
            analysis["result_dir"],
            job_id="perturb_running",
            job_name="Running Gene Perturbation",
            parent_job_id="analyze_explorer",
            iteration=1,
            status="running",
            log_text="[UNAGI-CUSTOM-PERTURB] stage=run status=START\n",
            finished_at=None,
        )
        response = await client.get(
            "/partials/analysis/analyze_explorer/explorer",
            params={
                "mode": "perturbation",
                "pert_panel_mode": "custom",
                "selected_custom_job_id": "perturb_running",
            },
        )
        assert response.status_code == 200
        assert "Still running." in response.text

        response = await client.get(
            "/api/analysis/analyze_explorer/perturbations/suggest",
            params={"kind": "pathway", "q": "patha"},
        )
        assert response.status_code == 200
        suggest_payload = response.json()
        assert suggest_payload["suggestions"][0]["value"] == "PathA"
        assert suggest_payload["suggestions"][0]["fill_gene_text"] == "GENE0, GENE1"


@pytest.mark.anyio
async def test_analysis_visualization_page_shows_generation_notice(tmp_path, monkeypatch):
    _patch_monitor_telemetry(monkeypatch)
    skill_root, repo_root, _ = create_monitor_workspace(tmp_path)
    train = create_train_job(
        skill_root,
        job_id="train_visualizing",
        job_name="Visualizing Training Job",
        status="succeeded",
        finished_at="2026-03-06T10:30:00+00:00",
        log_text="Iteration 0 training is done!\n",
    )
    analysis = create_analysis_job(
        train["job_dir"],
        job_id="analyze_visualizing",
        job_name="Visualizing Analysis Job",
        parent_job_id="train_visualizing",
        iteration=0,
        status="running",
        log_text="[UNAGI-ANALYSIS] stage=generate_visualizations status=START\n",
    )
    (analysis["result_dir"] / "visualization_status.json").write_text(
        json.dumps({"status": "running"}),
        encoding="utf-8",
    )

    app = create_monitor_app(skill_root=skill_root, repo_root=repo_root, skill_dir=SKILL_DIR)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/analysis/analyze_visualizing/visualizations")
        assert response.status_code == 200
        assert "Visualization generation is in progress" in response.text
        assert "Regenerate Visualizations" not in response.text


@pytest.mark.anyio
async def test_umap_api_switches_between_density_and_points_and_supports_selection_summary(tmp_path, monkeypatch):
    _patch_monitor_telemetry(monkeypatch)
    skill_root, repo_root, _ = create_monitor_workspace(tmp_path)
    train = create_train_job(
        skill_root,
        job_id="train_large_umap",
        job_name="Large UMAP Training Job",
        status="succeeded",
        finished_at="2026-03-06T10:30:00+00:00",
        log_text="Iteration 0 training is done!\n",
    )
    analysis = create_analysis_job(
        train["job_dir"],
        job_id="analyze_large_umap",
        job_name="Large UMAP Analysis Job",
        parent_job_id="train_large_umap",
        iteration=0,
        status="succeeded",
        log_text="[ANALYSIS] DONE\n",
    )
    write_analysis_dataset(
        analysis["result_dir"],
        stage_labels=("0",),
        cells_per_stage=60_000,
    )

    cache = UmapViewportCache(skill_root=skill_root)
    stage_shard = cache.get_stage_shard(analysis["result_dir"], stage="0")
    focus_x = float(stage_shard.coords[0, 0])
    focus_y = float(stage_shard.coords[0, 1])

    app = create_monitor_app(skill_root=skill_root, repo_root=repo_root, skill_dir=SKILL_DIR)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/analysis/analyze_large_umap/umap", params={"stage": "0"})
        assert response.status_code == 200
        density_payload = response.json()
        assert density_payload["render_mode"] == "sampled_points"
        assert density_payload["points"]
        assert density_payload["density"] is None

        response = await client.get(
            "/api/analysis/analyze_large_umap/umap",
            params={
                "stage": "0",
                "x_min": focus_x - 1e-6,
                "x_max": focus_x + 1e-6,
                "y_min": focus_y - 1e-6,
                "y_max": focus_y + 1e-6,
                "current_render_mode": "sampled_points",
            },
        )
        assert response.status_code == 200
        points_payload = response.json()
        assert points_payload["render_mode"] == "points"
        assert points_payload["point_count"] >= 1

        first_cell = points_payload["points"][0]["cell_id"]
        response = await client.post(
            "/api/analysis/analyze_large_umap/umap/selection-summary",
            json={
                "stage": "0",
                "color_by": "ident",
                "cell_ids": [first_cell],
                "render_mode": "points",
            },
        )
        assert response.status_code == 200
        summary = response.json()
        assert summary["selected_count"] == 1
        assert summary["columns"] == ["ident", "count"]

        response = await client.post(
            "/api/analysis/analyze_large_umap/umap/selection-summary",
            json={
                "stage": "0",
                "color_by": "ident",
                "cell_ids": [first_cell],
                "render_mode": "density",
            },
        )
        assert response.status_code == 400
        assert "point mode" in response.text

@pytest.mark.anyio
async def test_analysis_publication_pdf_routes_and_links(tmp_path, monkeypatch):
    _patch_monitor_telemetry(monkeypatch)
    skill_root, repo_root, _ = create_monitor_workspace(tmp_path)
    train = create_train_job(
        skill_root,
        job_id="train_pdf",
        job_name="PDF Training Job",
        status="succeeded",
        finished_at="2026-03-06T10:30:00+00:00",
        log_text="Iteration 1 training is done!\n",
    )
    analysis = create_analysis_job(
        train["job_dir"],
        job_id="analyze_pdf",
        job_name="PDF Analysis Job",
        parent_job_id="train_pdf",
        iteration=1,
        status="succeeded",
        with_visualization_manifest=True,
        log_text="[ANALYSIS] DONE\n",
    )
    write_analysis_dataset(analysis["result_dir"])

    visualizations_dir = analysis["result_dir"] / "visualizations"
    tables_dir = visualizations_dir / "tables"
    stage_png = visualizations_dir / "results_stage_composition.png"
    stage_table = tables_dir / "results_stage_composition_table.csv"
    stage_png.write_text("stub", encoding="utf-8")
    stage_table.write_text("stage,AT1\n0,0.4\n", encoding="utf-8")
    (visualizations_dir / "manifest.json").write_text(
        json.dumps(
            {
                "analysis_dir": str(analysis["result_dir"]),
                "figure_dir": str(visualizations_dir),
                "tables_dir": str(tables_dir),
                "successes": [
                    {
                        "name": "stage_composition",
                        "file": str(stage_png),
                        "table": str(stage_table),
                    }
                ],
                "failures": [],
                "counts": {"successes": 1, "failures": 0},
            }
        ),
        encoding="utf-8",
    )

    app = create_monitor_app(skill_root=skill_root, repo_root=repo_root, skill_dir=SKILL_DIR)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/analysis/analyze_pdf/visualizations")
        assert response.status_code == 200
        assert "/analysis/analyze_pdf/visualizations/stage_composition/publication.pdf" in response.text
        assert "Temporal Dynamics Overview" in response.text
        assert "Custom Perturbation" not in response.text
        assert 'data-collapsible-id="group-temporal_dynamics_overview"' not in response.text
        assert ">raw<" not in response.text
        assert ">browse<" not in response.text
        assert "<strong>file</strong>" not in response.text

        for figure_name in (
            "stage_umap_grid",
            "stage_composition",
            "pathway_ranked_bar",
            "drug_ranked_bar",
            "single_gene_ranked_bar",
            "gene_combo_ranked_bar",
            "two_drugs_ranked_bar",
            "dynamic_marker_trajectories",
            "dynamic_marker_heatmap",
            "hierarchical_marker_heatmap",
            "temporal_dynamics_graph",
        ):
            response = await client.get(f"/analysis/analyze_pdf/visualizations/{figure_name}/publication.pdf")
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("application/pdf")
            assert response.content.startswith(b"%PDF")

        response = await client.get(
            "/analysis/analyze_pdf/explorer/publication.pdf",
            params={
                "mode": "dynamic_markers",
                "figure_key": "trajectories",
                "dyn_track": "0-2-4",
                "dyn_directions": ["increasing"],
                "dyn_plot_kind": "trajectories",
            },
        )
        assert response.status_code == 200
        assert response.content.startswith(b"%PDF")

        response = await client.get(
            "/analysis/analyze_pdf/explorer/publication.pdf",
            params={
                "mode": "umap",
                "figure_key": "umap",
                "umap_stage": "1",
                "umap_color_by": "ident",
                "umap_gene": "GENE0",
            },
        )
        assert response.status_code == 200
        assert response.content.startswith(b"%PDF")
        assert "attachment;" in response.headers.get("content-disposition", "")


@pytest.mark.anyio
async def test_analysis_explorer_handles_missing_sections_and_scanned_roots(tmp_path, monkeypatch):
    _patch_monitor_telemetry(monkeypatch)
    skill_root, repo_root, _ = create_monitor_workspace(tmp_path)
    external_root = tmp_path / "external_workspace"
    external_train = create_train_job(
        external_root,
        job_id="train_external_explorer",
        job_name="External Explorer Training",
        status="succeeded",
        finished_at="2026-03-06T10:30:00+00:00",
        log_text="done\n",
    )
    external_analysis = create_analysis_job(
        external_train["job_dir"],
        job_id="analyze_external_explorer",
        job_name="External Explorer Analysis",
        parent_job_id="train_external_explorer",
        iteration=0,
        status="succeeded",
        log_text="[ANALYSIS] DONE\n",
    )
    write_analysis_dataset(
        external_analysis["result_dir"],
        include_hierarchical=False,
        include_dynamic=False,
        include_perturbation=False,
    )
    external_jobs_root = external_root / "outputs" / "skill_jobs"

    app = create_monitor_app(
        skill_root=skill_root,
        repo_root=repo_root,
        skill_dir=SKILL_DIR,
        job_roots=[external_jobs_root],
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/partials/analysis/analyze_external_explorer/explorer")
        assert response.status_code == 200
        assert "Interactive Explorer" not in response.text

        response = await client.post(
            "/partials/analysis/analyze_external_explorer/explorer/render",
            data={"mode": "hierarchical"},
        )
        assert response.status_code == 200
        assert "ignore_hcmarkers" in response.text
