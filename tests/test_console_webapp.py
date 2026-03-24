from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path

import httpx
import pytest

import UNAGI.monitoring.service as service_module
import UNAGI.monitoring.webapp as webapp_module
from UNAGI.monitoring.webapp import create_monitor_app

from .interactive_analysis_test_helpers import write_analysis_dataset
from .monitor_test_helpers import (
    SKILL_DIR,
    create_analysis_job,
    create_custom_perturbation_job,
    create_monitor_workspace,
    create_train_job,
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
            "gpus": [],
            "gpu_error": None,
            "gpu_pid_map": {},
        },
    )
    monkeypatch.setattr(service_module, "collect_process_metrics", lambda pid, gpu_pid_map=None: None)


def test_visualization_context_hides_unnecessary_tables_and_paths(tmp_path):
    output_dir = Path(tmp_path) / "analysis_results"
    visualizations_dir = output_dir / "visualizations"
    tables_dir = visualizations_dir / "tables"
    idrem_dir = output_dir / "idrem"
    tables_dir.mkdir(parents=True, exist_ok=True)
    idrem_dir.mkdir(parents=True, exist_ok=True)

    ranked_bar = visualizations_dir / "results_pathway_ranked_bar.png"
    pert_table = tables_dir / "results_pathway_perturbation_table.csv"
    stage_comp = visualizations_dir / "results_stage_composition.png"
    stage_comp_table = tables_dir / "results_stage_composition_table.csv"
    hier_heatmap = visualizations_dir / "results_hierarchical_marker_heatmap_stage0_cluster0_level0.png"
    hier_table = tables_dir / "results_hierarchical_marker_table_stage0_cluster0_level0.csv"
    hier_matrix = tables_dir / "results_hierarchical_marker_matrix_stage0_cluster0_level0.csv"
    dynamic_plot = visualizations_dir / "results_dynamic_marker_trajectories.png"
    dynamic_table = tables_dir / "results_dynamic_marker_trajectories_long.csv"
    graph_png = visualizations_dir / "results_temporal_dynamics_graph.png"
    graph_nodes = tables_dir / "results_temporal_dynamics_nodes.csv"
    graph_edges = tables_dir / "results_temporal_dynamics_edges.csv"
    idrem_html = idrem_dir / "report.html"
    for path in (
        ranked_bar,
        pert_table,
        stage_comp,
        stage_comp_table,
        hier_heatmap,
        hier_table,
        hier_matrix,
        dynamic_plot,
        dynamic_table,
        graph_png,
        graph_nodes,
        graph_edges,
    ):
        path.write_text("stub", encoding="utf-8")
    idrem_html.write_text("<html><body>iDREM</body></html>", encoding="utf-8")

    job = {
        "job_id": "analyze_ctx",
        "output_path": str(output_dir),
        "visualizations": {
            "manifest": {
                "successes": [
                    {"name": "pathway_perturbation_table", "table": str(pert_table)},
                    {"name": "pathway_ranked_bar", "file": str(ranked_bar)},
                    {"name": "stage_composition", "file": str(stage_comp), "table": str(stage_comp_table)},
                    {
                        "name": "hierarchical_marker_heatmap",
                        "file": str(hier_heatmap),
                        "table": str(hier_table),
                        "matrix": str(hier_matrix),
                    },
                    {"name": "dynamic_marker_trajectories", "file": str(dynamic_plot), "table": str(dynamic_table)},
                    {"name": "temporal_dynamics_graph", "file": str(graph_png), "nodes": str(graph_nodes), "edges": str(graph_edges)},
                ],
                "failures": [],
                "counts": {"successes": 4, "failures": 0},
            }
        },
    }

    context = webapp_module._visualization_context(job)
    names = [entry["name"] for entry in context["entries"]]
    assert "pathway_perturbation_table" not in names
    assert "pathway_ranked_bar" in names

    ranked_entry = next(entry for entry in context["entries"] if entry["name"] == "pathway_ranked_bar")
    assert ranked_entry["artifacts"][0]["downloads"][0]["path"] == str(pert_table)
    assert ranked_entry["artifacts"][0]["publication_download"].endswith("/analysis/%s/visualizations/pathway_ranked_bar/publication.pdf" % job["job_id"])

    stage_comp_entry = next(entry for entry in context["entries"] if entry["name"] == "stage_composition")
    assert [artifact["label"] for artifact in stage_comp_entry["artifacts"]] == ["file"]
    assert stage_comp_entry["artifacts"][0]["downloads"] == []

    hier_entry = next(entry for entry in context["entries"] if entry["name"] == "hierarchical_marker_heatmap")
    assert [artifact["label"] for artifact in hier_entry["artifacts"]] == ["file"]
    assert [download["label"] for download in hier_entry["artifacts"][0]["downloads"]] == ["table", "matrix"]

    dynamic_entry = next(entry for entry in context["entries"] if entry["name"] == "dynamic_marker_trajectories")
    assert [artifact["label"] for artifact in dynamic_entry["artifacts"]] == ["file"]

    graph_entry = next(entry for entry in context["entries"] if entry["name"] == "temporal_dynamics_graph")
    assert [artifact["label"] for artifact in graph_entry["artifacts"]] == ["file"]
    assert context["idrem_files"][0]["name"] == "report.html"

    group_labels = [group["label"] for group in context["groups"]]
    assert "Dynamic markers" in group_labels
    assert "Hierarchical markers" in group_labels
    assert [entry["name"] for entry in context["temporal_overview"]] == [
        "temporal_dynamics_graph",
        "stage_composition",
    ]

    dynamic_group = next(group for group in context["groups"] if group["label"] == "Dynamic markers")
    assert [entry["name"] for entry in dynamic_group["entries"]] == ["dynamic_marker_trajectories"]

    hier_group = next(group for group in context["groups"] if group["label"] == "Hierarchical markers")
    assert [entry["name"] for entry in hier_group["entries"]] == ["hierarchical_marker_heatmap"]


@pytest.mark.anyio
async def test_console_routes_support_forms_actions_visualizations_and_path_picker(tmp_path, monkeypatch):
    _patch_monitor_telemetry(monkeypatch)
    skill_root, repo_root, _ = create_monitor_workspace(tmp_path)

    data_root = skill_root / "datasets"
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "demo.h5ad").write_text("stub", encoding="utf-8")

    train = create_train_job(
        skill_root,
        job_id="train_001",
        job_name="Console Training Job",
        status="succeeded",
        finished_at="2026-03-06T10:30:00+00:00",
        finished_iterations=[1],
        log_text="Iteration 1 training is done!\n",
    )
    create_train_job(
        skill_root,
        job_id="train_active",
        job_name="Active Training Job",
        status="running",
        log_text="[UNAGI-TRAIN] iteration=0 max_iter=2 stage=training_model status=START\n",
    )
    analysis = create_analysis_job(
        train["job_dir"],
        job_id="analyze_001",
        job_name="Console Analysis Job",
        parent_job_id="train_001",
        iteration=1,
        status="succeeded",
        with_visualization_manifest=True,
        log_text="[ANALYSIS] DONE\n",
    )
    write_analysis_dataset(analysis["result_dir"])

    monkeypatch.setattr(
        webapp_module,
        "run_workflow_from_payload",
        lambda payload, validate_only=False, persist_updates=False: {
            "data_dir": str(skill_root / "normalized_data"),
            "idrem_dir": str(repo_root / "idrem"),
        },
    )
    monkeypatch.setattr(
        job_backend,
        "create_job_from_config_payload",
        lambda **kwargs: {"job_id": "train_new", "job_name": kwargs.get("job_name") or "train_new"},
    )
    monkeypatch.setattr(
        job_backend,
        "create_analysis_job_from_params",
        lambda **kwargs: {"job_id": "analyze_new", "job_name": kwargs.get("job_name") or "analyze_new"},
    )
    monkeypatch.setattr(
        job_backend,
        "create_custom_perturbation_job_from_payload",
        lambda **kwargs: {
            "job_id": "perturb_new",
            "job_name": kwargs.get("job_name") or "perturb_new",
            "job_kind": "custom_perturbation",
        },
    )
    monkeypatch.setattr(job_backend, "cancel_job", lambda **kwargs: {"status": "cancel_requested", "job_id": kwargs["job_id"]})
    monkeypatch.setattr(job_backend, "resume_job", lambda **kwargs: {"status": "resubmitted", "job_id": kwargs["job_id"]})
    def _delete_job_from_file(**kwargs):
        if "train_active" in str(kwargs["job_file_path"]):
            raise ValueError("Job train_active must be in a terminal state before deletion; current status=running")
        job_id = str(kwargs["job_file_path"]).split("/")[-2]
        return {"status": "deleted", "job_id": job_id, "deleted_paths": [f"/tmp/{job_id}"], "warnings": []}
    monkeypatch.setattr(job_backend, "delete_job_from_file", _delete_job_from_file)
    monkeypatch.setattr(
        job_backend,
        "delete_jobs_from_files",
        lambda **kwargs: {
            "status": "ok",
            "deleted": [{"job_id": "train_001", "deleted_paths": ["/tmp/train_001"], "warnings": []}],
            "blocked": [{"job_id": "train_active", "error": "terminal state required"}],
            "failed": [],
        },
    )
    monkeypatch.setattr(
        webapp_module,
        "generate_analysis_visualizations",
        lambda analysis_dir: {
            "analysis_dir": str(analysis_dir),
            "successes": [],
            "failures": [],
            "counts": {"successes": 0, "failures": 0},
        },
    )

    app = create_monitor_app(skill_root=skill_root, repo_root=repo_root, skill_dir=SKILL_DIR, data_roots=[data_root])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/experiments/new")
        assert response.status_code == 200
        assert "Create A Durable Training Job" in response.text

        response = await client.post(
            "/api/training/validate",
            data={
                "template_path": str(repo_root / "configs" / "unagi_workflow.example.run.yaml"),
                "data.path": str(data_root / "demo.h5ad"),
                "data.stage_key": "stage",
                "data.total_stage": "2",
                "data.label_key": "name.simple",
                "training.task": "train",
                "training.dist": "zinb",
                "training.GPU": "false",
                "training.max_iter": "2",
                "advanced_yaml": "",
            },
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True

        response = await client.post(
            "/api/training",
            data={
                "job_name": "From Console",
                "template_path": str(repo_root / "configs" / "unagi_workflow.example.run.yaml"),
                "data.path": str(data_root / "demo.h5ad"),
                "data.stage_key": "stage",
                "data.total_stage": "2",
                "data.label_key": "name.simple",
                "training.task": "train",
                "training.dist": "zinb",
                "training.GPU": "false",
                "training.max_iter": "2",
                "advanced_yaml": "",
            },
        )
        assert response.status_code == 201
        assert response.json()["job_id"] == "train_new"

        response = await client.get("/jobs/train_001/analysis/new", params={"iteration": 1})
        assert response.status_code == 200
        assert "Launch Analysis For Iteration 1" in response.text
        assert "different random backgrounds" in response.text

        response = await client.get("/jobs/train_001")
        assert response.status_code == 200
        assert "Delete Entire Training Task" in response.text
        assert "nested analysis jobs and their outputs" in response.text

        response = await client.get("/jobs/train_active")
        assert response.status_code == 200
        assert "Delete Disabled While Active" in response.text

        response = await client.post(
            "/api/analysis",
            data={
                "training_job_id": "train_001",
                "iteration": "1",
                "template_path": str(repo_root / "configs" / "unagi_analysis_template.yaml"),
                "run_perturbation": "false",
                "direction_mode": "one_sided",
                "advanced_yaml": "",
            },
        )
        assert response.status_code == 201
        assert response.json()["job_id"] == "analyze_new"

        response = await client.post("/api/jobs/train_001/cancel")
        assert response.status_code == 200
        assert response.json()["status"] == "cancel_requested"

        response = await client.post("/api/jobs/train_001/resume", data={})
        assert response.status_code == 200
        assert response.json()["status"] == "resubmitted"

        response = await client.post("/api/jobs/train_001/delete")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

        response = await client.post("/api/jobs/train_active/delete")
        assert response.status_code == 400
        assert response.json()["status"] == "blocked"

        response = await client.get("/partials/dashboard")
        assert response.status_code == 200
        assert "Training Runs" in response.text
        assert "Analysis Jobs" in response.text
        assert "Delete Selected" in response.text
        assert 'name="job_id"' in response.text

        response = await client.post(
            "/api/jobs/delete",
            data={"job_id": ["train_001", "train_active"]},
        )
        assert response.status_code == 200
        bulk = response.json()
        assert bulk["deleted"][0]["job_id"] == "train_001"
        assert bulk["blocked"][0]["job_id"] == "train_active"

        response = await client.get("/analysis/analyze_001/visualizations")
        assert response.status_code == 200
        assert "Custom Perturbation" not in response.text
        assert "Visualization Modules" in response.text
        assert "Manifest" not in response.text
        assert "report.html" in response.text
        assert 'data-collapsible-group="visualization-modules-analyze_001"' in response.text
        assert 'data-collapsible-id="idrem"' in response.text
        assert 'data-collapsible-id="failures"' not in response.text
        assert "module-tab-button" not in response.text

        response = await client.post(
            "/api/analysis/analyze_001/perturbations/validate",
            data={
                "mode": "gene",
                "scope_kind": "track",
                "track_key": "0-2-4",
                "perturb_change": "0.5",
                "direction_mode": "two_sided",
                "execution_mode": "auto",
                "signed_gene_text": "gene0:+, gene1:-",
            },
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True
        assert response.json()["normalized"]["target_spec"]["signed_genes"] == ["GENE0:+", "GENE1:-"]

        response = await client.post(
            "/api/analysis/analyze_001/perturbations",
            data={
                "job_name": "Queued perturbation",
                "mode": "gene",
                "scope_kind": "overall",
                "perturb_change": "0.5",
                "direction_mode": "two_sided",
                "execution_mode": "auto",
                "signed_gene_text": "gene0:+",
            },
        )
        assert response.status_code == 201
        assert response.json()["job_id"].startswith("perturb_")

        response = await client.get(
            "/partials/path-picker",
            params={"target": "data.path", "kind": "training_data", "path": str(data_root)},
        )
        assert response.status_code == 200
        assert "demo.h5ad" in response.text
        assert "Use" in response.text

        response = await client.post("/api/analysis/analyze_001/visualizations/generate")
        assert response.status_code == 202
        task_id = response.json()["task_id"]

        task_payload = None
        for _ in range(10):
            await asyncio.sleep(0.05)
            poll = await client.get(f"/api/direct-tasks/{task_id}")
            assert poll.status_code == 200
            task_payload = poll.json()
            if task_payload["status"] == "succeeded":
                break
        assert task_payload is not None
        assert task_payload["status"] == "succeeded"

        response = await client.get("/browse", params={"path": "/etc/passwd"})
        assert response.status_code == 404


@pytest.mark.anyio
async def test_custom_perturbation_result_routes_render_and_serialize(tmp_path, monkeypatch):
    _patch_monitor_telemetry(monkeypatch)
    skill_root, repo_root, _ = create_monitor_workspace(tmp_path)
    train = create_train_job(
        skill_root,
        job_id="train_parent",
        job_name="Training Parent",
        status="succeeded",
        finished_iterations=[1],
        log_text="Iteration 1 training is done!\n",
    )
    analysis = create_analysis_job(
        train["job_dir"],
        job_id="analyze_parent",
        job_name="Analysis Parent",
        parent_job_id="train_parent",
        iteration=1,
        status="succeeded",
        with_visualization_manifest=True,
        log_text="[ANALYSIS] DONE\n",
    )
    write_analysis_dataset(analysis["result_dir"])
    create_custom_perturbation_job(
        analysis["job_dir"],
        analysis["result_dir"],
        job_id="perturb_done",
        job_name="Online Gene Perturbation",
        parent_job_id="analyze_parent",
        iteration=1,
        status="succeeded",
        log_text="[UNAGI-CUSTOM-PERTURB] stage=save status=DONE\n",
    )

    app = create_monitor_app(skill_root=skill_root, repo_root=repo_root, skill_dir=SKILL_DIR)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/analysis/analyze_parent/perturbations/perturb_done")
        assert response.status_code == 200
        assert "Custom Perturbation" in response.text
        assert "Custom Perturbation Result" in response.text
        assert "EPAS1" in response.text

        response = await client.get("/api/analysis/analyze_parent/perturbations/perturb_done")
        assert response.status_code == 200
        payload = response.json()
        assert payload["job"]["job_id"] == "perturb_done"
        assert payload["result"]["metadata"]["track_label"] == "Overall"


@pytest.mark.anyio
async def test_custom_perturbation_result_routes_flag_stale_gene_bundles(tmp_path, monkeypatch):
    _patch_monitor_telemetry(monkeypatch)
    skill_root, repo_root, _ = create_monitor_workspace(tmp_path)
    train = create_train_job(
        skill_root,
        job_id="train_parent",
        job_name="Training Parent",
        status="succeeded",
        finished_iterations=[1],
        log_text="Iteration 1 training is done!\n",
    )
    analysis = create_analysis_job(
        train["job_dir"],
        job_id="analyze_parent",
        job_name="Analysis Parent",
        parent_job_id="train_parent",
        iteration=1,
        status="succeeded",
        with_visualization_manifest=True,
        log_text="[ANALYSIS] DONE\n",
    )
    write_analysis_dataset(analysis["result_dir"])
    stale = create_custom_perturbation_job(
        analysis["job_dir"],
        analysis["result_dir"],
        job_id="perturb_stale",
        job_name="Old Gene Perturbation",
        parent_job_id="analyze_parent",
        iteration=1,
        status="succeeded",
        log_text="[UNAGI-CUSTOM-PERTURB] stage=save status=DONE\n",
    )
    summary_path = stale["result_dir"] / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["preview"]["rows"] = [["bg_0", "3.2"]]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    app = create_monitor_app(skill_root=skill_root, repo_root=repo_root, skill_dir=SKILL_DIR)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/analysis/analyze_parent/perturbations/perturb_stale")
        assert response.status_code == 200
        assert "needs to be rerun" in response.text

        response = await client.get("/api/analysis/analyze_parent/perturbations/perturb_stale")
        assert response.status_code == 409
        assert "needs to be rerun" in response.text
