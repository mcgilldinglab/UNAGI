from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = REPO_ROOT / "skills" / "unagi-workflow-orchestrator"


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def create_monitor_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    skill_root = tmp_path / "UNAGI_skill_workspace"
    repo_root = tmp_path / "UNAGI_repo"
    (repo_root / "UNAGI").mkdir(parents=True, exist_ok=True)
    (repo_root / "UNAGI" / "cli.py").write_text("# stub\n", encoding="utf-8")
    (repo_root / "UNAGI" / "data").mkdir(parents=True, exist_ok=True)
    (repo_root / "UNAGI" / "data" / "cmap_drug_target_genes.npy").write_text("stub", encoding="utf-8")
    (repo_root / "configs").mkdir(parents=True, exist_ok=True)
    write_yaml(
        repo_root / "configs" / "unagi_workflow.example.run.yaml",
        {
            "data": {"path": "/tmp/example.h5ad", "stage_key": "stage", "total_stage": 2, "label_key": "name.simple"},
            "training": {
                "task": "train",
                "dist": "zinb",
                "GPU": False,
                "epoch_initial": 1,
                "epoch_iter": 2,
                "max_iter": 2,
                "BATCHSIZE": 32,
                "adversarial": True,
                "GCN": True,
            },
            "run": {"CPO": True, "connect_edges_cutoff": 0.05},
        },
    )
    write_yaml(
        repo_root / "configs" / "unagi_analysis_template.yaml",
        {
            "random_background_sampling_times": 20,
            "run_perturbation": True,
            "direction_mode": "two_sided",
            "overall_perturbation_analysis": True,
            "perturbed_tracks": "all",
            "ignore_pathway_perturabtion": False,
            "ignore_drug_perturabtion": False,
            "centroid": False,
            "ignore_hcmarkers": False,
            "ignore_dynamic_markers": False,
        },
    )
    return skill_root, repo_root, SKILL_DIR


def create_train_job(
    skill_root: Path,
    *,
    job_id: str,
    job_name: str,
    status: str,
    log_text: str,
    config: dict[str, Any] | None = None,
    started_at: str = "2026-03-06T10:00:00+00:00",
    updated_at: str = "2026-03-06T10:05:00+00:00",
    finished_at: str | None = None,
    child_pid: int | None = 4321,
    stale_paths: bool = False,
    finished_iterations: list[int] | None = None,
) -> dict[str, Path]:
    job_dir = skill_root / "outputs" / "skill_jobs" / job_id
    config_path = write_yaml(
        job_dir / "config.yaml",
        config or {"training": {"max_iter": 2, "epoch_initial": 1, "epoch_iter": 5}},
    )
    log_path = job_dir / "training.log"
    log_path.write_text(log_text, encoding="utf-8")
    output_path = job_dir / "input"
    (output_path / "model_save").mkdir(parents=True, exist_ok=True)
    (job_dir / "analysis").mkdir(parents=True, exist_ok=True)
    for iteration in finished_iterations or []:
        dataset_path = output_path / str(iteration) / "stagedata" / "dataset.h5ad"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text("stub", encoding="utf-8")

    payload = {
        "job_id": job_id,
        "job_name": job_name,
        "job_kind": "train",
        "status": status,
        "created_at": started_at,
        "updated_at": updated_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "attempts": 1,
        "max_retries": 0,
        "child_pid": child_pid,
        "config_path": str(job_dir / "stale-config.yaml") if stale_paths else str(config_path),
        "log_path": str(job_dir / "stale-training.log") if stale_paths else str(log_path),
    }
    write_json(job_dir / "job.json", payload)
    return {
        "job_dir": job_dir,
        "config_path": config_path,
        "log_path": log_path,
        "output_path": output_path,
        "job_json": job_dir / "job.json",
    }


def create_analysis_job(
    train_job_dir: Path,
    *,
    job_id: str,
    job_name: str,
    parent_job_id: str,
    iteration: int,
    status: str,
    log_text: str,
    config: dict[str, Any] | None = None,
    started_at: str = "2026-03-06T11:00:00+00:00",
    updated_at: str = "2026-03-06T11:20:00+00:00",
    finished_at: str | None = "2026-03-06T11:30:00+00:00",
    child_pid: int | None = None,
    stale_paths: bool = False,
    with_visualization_manifest: bool = False,
) -> dict[str, Path]:
    job_dir = train_job_dir / "analysis_jobs" / job_id
    result_dir = train_job_dir / "analysis" / f"iter{iteration}" / job_id / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "model_save").mkdir(parents=True, exist_ok=True)
    (result_dir / "idrem").mkdir(parents=True, exist_ok=True)
    if with_visualization_manifest:
        visualizations_dir = result_dir / "visualizations"
        tables_dir = visualizations_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "analysis_dir": str(result_dir),
            "figure_dir": str(visualizations_dir),
            "tables_dir": str(tables_dir),
            "successes": [],
            "failures": [],
            "counts": {"successes": 0, "failures": 0},
        }
        write_json(visualizations_dir / "manifest.json", manifest)
        (result_dir / "idrem" / "report.html").write_text("<html><body>iDREM</body></html>", encoding="utf-8")

    config_path = write_yaml(
        job_dir / "analysis_params.yaml",
        config
        or {
            "iteration": iteration,
            "run_perturbation": True,
            "ignore_hcmarkers": False,
            "ignore_dynamic_markers": False,
            "ignore_pathway_perturabtion": False,
            "ignore_drug_perturabtion": False,
            "direction_mode": "two_sided",
            "centroid": False,
        },
    )
    log_path = job_dir / "analysis.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log_text, encoding="utf-8")

    payload = {
        "job_id": job_id,
        "job_name": job_name,
        "job_kind": "analysis",
        "status": status,
        "created_at": started_at,
        "updated_at": updated_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "attempts": 1,
        "max_retries": 0,
        "analysis_iteration": iteration,
        "parent_job_id": parent_job_id,
        "child_pid": child_pid,
        "config_path": str(job_dir / "stale-analysis.yaml") if stale_paths else str(config_path),
        "log_path": str(job_dir / "stale-analysis.log") if stale_paths else str(log_path),
        "analysis_output_dir": str(result_dir.parent / "stale-results") if stale_paths else str(result_dir),
    }
    write_json(job_dir / "job.json", payload)
    return {
        "job_dir": job_dir,
        "config_path": config_path,
        "log_path": log_path,
        "result_dir": result_dir,
        "job_json": job_dir / "job.json",
    }


def create_custom_perturbation_job(
    analysis_job_dir: Path,
    analysis_result_dir: Path,
    *,
    job_id: str,
    job_name: str,
    parent_job_id: str,
    iteration: int,
    status: str,
    log_text: str,
    request_payload: dict[str, Any] | None = None,
    started_at: str = "2026-03-06T11:40:00+00:00",
    updated_at: str = "2026-03-06T11:45:00+00:00",
    finished_at: str | None = "2026-03-06T11:50:00+00:00",
) -> dict[str, Path]:
    job_dir = analysis_job_dir / "perturbation_jobs" / job_id
    result_dir = analysis_result_dir / "custom_perturbations" / job_id
    result_dir.mkdir(parents=True, exist_ok=True)

    request_path = write_json(
        job_dir / "request.json",
        request_payload
        or {
            "mode": "gene",
            "scope_kind": "overall",
            "analysis_iteration": iteration,
            "analysis_job_id": parent_job_id,
            "analysis_dir": str(analysis_result_dir),
            "target_spec": {"genes": ["EPAS1", "THBD"]},
            "execution_mode": "auto",
            "requested_execution_mode": "auto",
            "direction_mode": "two_sided",
            "perturb_change": 0.5,
        },
    )
    log_path = job_dir / "perturbation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log_text, encoding="utf-8")
    summary_path = write_json(
        result_dir / "summary.json",
        {
            "request": {
                "mode": "gene",
                "scope_kind": "overall",
                "target_spec": {"genes": ["EPAS1", "THBD"]},
            },
            "metadata": {
                "resolved_track_key": "overall",
                "track_label": "Overall",
                "analysis_iteration": iteration,
                "execution_mode": "auto",
                "direction_mode": "two_sided",
                "scope_kind": "overall",
            },
            "table_csv_path": str(result_dir / "result_table.csv"),
            "table_json_path": str(result_dir / "result_table.json"),
            "plot_png_path": str(result_dir / "ranked_bar.png"),
            "plot_pdf_path": str(result_dir / "ranked_bar.pdf"),
            "preview": {
                "columns": ["gene", "perturbation score"],
                "rows": [["EPAS1", "3.2"], ["THBD", "2.6"]],
            },
            "row_count": 2,
        },
    )
    (result_dir / "result_table.csv").write_text("gene,perturbation score\nEPAS1,3.2\nTHBD,2.6\n", encoding="utf-8")
    (result_dir / "result_table.json").write_text("[]", encoding="utf-8")
    (result_dir / "ranked_bar.png").write_text("stub", encoding="utf-8")
    (result_dir / "ranked_bar.pdf").write_text("stub", encoding="utf-8")

    payload = {
        "job_id": job_id,
        "job_name": job_name,
        "job_kind": "custom_perturbation",
        "status": status,
        "created_at": started_at,
        "updated_at": updated_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "attempts": 1,
        "max_retries": 0,
        "analysis_iteration": iteration,
        "parent_job_id": parent_job_id,
        "config_path": str(request_path),
        "log_path": str(log_path),
        "analysis_output_dir": str(result_dir),
        "queue_key": "gpu:0",
        "execution_mode": "auto",
        "scope_kind": "overall",
        "perturbation_mode": "gene",
        "result_summary_path": str(summary_path),
    }
    write_json(job_dir / "job.json", payload)
    return {
        "job_dir": job_dir,
        "config_path": request_path,
        "log_path": log_path,
        "result_dir": result_dir,
        "summary_path": summary_path,
        "job_json": job_dir / "job.json",
    }


def write_mock_idrem_track(result_dir: Path, *, track_key: str = "0-2-4") -> Path:
    track_dir = result_dir / "idrem" / f"{track_key}.txt_viz"
    (track_dir / "d3").mkdir(parents=True, exist_ok=True)
    (track_dir / "d3.chart.sankey").mkdir(parents=True, exist_ok=True)
    nodes = [
        {
            "parent": -1,
            "children": [1, 2, -1],
            "nodeID": 0,
            "nodetime": "stage0",
            "nodeMean": 0.0,
            "nodeSigma": 0.0,
        },
        {
            "parent": 0,
            "children": [-1, -1, -1],
            "nodeID": 1,
            "nodetime": "stage1",
            "nodeMean": 0.65,
            "nodeSigma": 0.12,
            "genesInNode": [True, False, True],
            "ETF": [
                ["TFAP2A 1", "24", "24", "7", "1.5", "0.4", "0.002"],
                ["MIR-21 1", "16", "16", "5", "1.1", "0.5", "0.006"],
            ],
            "goNode": [
                ["GO:0001525", "angiogenesis", "p-value: 1e-4", "corrected p-value:0.002"],
                ["GO:0000000", "filtered term", "p-value: 0.10", "corrected p-value:0.20"],
            ],
        },
        {
            "parent": 0,
            "children": [-1, -1, -1],
            "nodeID": 2,
            "nodetime": "stage1",
            "nodeMean": 0.18,
            "nodeSigma": 0.08,
            "genesInNode": [False, True, False],
            "ETF": [["SOX9 1", "12", "12", "3", "1.0", "0.7", "0.01"]],
            "goNode": [["GO:0048514", "blood vessel morphogenesis", "p-value: 5e-4", "corrected p-value:0.01"]],
        },
    ]
    payload = [
        nodes,
        None,
        [],
        ["GENE0", "GENE1", "GENE2"],
        ["TFAP2A", "SOX9", "MIR-21"],
        [[0.0, 0.8], [0.0, 0.3], [0.0, 0.6]],
        None,
        None,
        [
            ["gene", "stage0", "stage1"],
            ["GENE0", "0.0", "0.8"],
            ["GENE1", "0.0", "0.3"],
            ["GENE2", "0.0", "0.6"],
        ],
        None,
        None,
    ]
    (track_dir / "DREM.json").write_text(f"data={json.dumps(payload)};", encoding="utf-8")
    (track_dir / "parseJSON.js").write_text("// mock iDREM parser\n", encoding="utf-8")
    (track_dir / "idrem_result.html").write_text("<html><body>mock iDREM</body></html>", encoding="utf-8")
    return track_dir


def fake_toppgene_enrichment_table() -> dict[str, Any]:
    return {
        "columns": ["Category", "Term ID", "Term Name", "corrected p-value", "p-value", "Genes in query"],
        "rows": [
            ["GO Biological Process", "GO:0001525", "angiogenesis", "2.0000e-03", "1.0000e-04", "GENE0"],
            ["Pathway", "R-HSA-12345", "VEGF signaling", "4.0000e-02", "1.5000e-03", "GENE0, GENE2"],
        ],
        "row_count": 2,
        "hover_texts": [
            "Matched genes: GENE0",
            "Matched genes: GENE0, GENE2",
        ],
        "message": "",
        "source": "ToppGene",
    }
