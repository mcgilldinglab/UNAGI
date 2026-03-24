from __future__ import annotations

from pathlib import Path

import pytest

import UNAGI.monitoring.service as service_module
from UNAGI.monitoring.service import JobMonitorService

from .monitor_test_helpers import (
    create_analysis_job,
    create_custom_perturbation_job,
    create_monitor_workspace,
    create_train_job,
)


@pytest.fixture
def quiet_monitor(monkeypatch):
    monkeypatch.setattr(
        service_module,
        "collect_system_metrics",
        lambda skill_root: {
            "cpu_percent": 12.5,
            "memory": {"percent": 34.0},
            "disk": {"percent": 56.0},
            "gpus": [],
            "gpu_error": "GPU telemetry unavailable: nvidia-smi not found",
            "gpu_pid_map": {},
        },
    )
    monkeypatch.setattr(service_module, "collect_process_metrics", lambda pid, gpu_pid_map=None: None)


def test_list_jobs_discovers_train_and_analysis_and_links_parent_child(tmp_path, monkeypatch, quiet_monitor):
    skill_root, repo_root, _skill_dir = create_monitor_workspace(tmp_path)
    train = create_train_job(
        skill_root,
        job_id="train_001",
        job_name="Example Training",
        status="running",
        finished_iterations=[0, 1],
        stale_paths=True,
        log_text=(
            "[UNAGI] iteration 0: loading staged data\n"
            "[UNAGI] iteration 0: training model\n"
            "[TRAIN] iteration 0 epoch 1/5 loss=9.5000\n"
        ),
    )
    analysis = create_analysis_job(
        train["job_dir"],
        job_id="analyze_001",
        job_name="Iteration 0 Analysis",
        parent_job_id="train_001",
        iteration=0,
        status="succeeded",
        with_visualization_manifest=True,
        stale_paths=True,
        log_text=(
            "[UNAGI-ANALYSIS] start analyse iteration=0, run_perturbation=True, centroid=False, "
            "perturb_change=0.5, direction_mode=two_sided\n"
            "[UNAGI-ANALYSIS] stage=hierarchical_markers status=START\n"
            "[UNAGI-ANALYSIS] stage=hierarchical_markers status=DONE\n"
            "[ANALYSIS] DONE\n"
        ),
    )

    def fake_process(pid, gpu_pid_map=None):
        if pid == 4321:
            return {"pid": 4321, "cpu_percent": 11.0, "memory_rss_bytes": 2048, "status": "running"}
        return None

    monkeypatch.setattr(service_module, "collect_process_metrics", fake_process)
    service = JobMonitorService(skill_root=skill_root, repo_root=repo_root)

    jobs = service.list_jobs()

    assert [job["job_id"] for job in jobs] == ["train_001", "analyze_001"]

    train_job = jobs[0]
    analysis_job = jobs[1]

    assert train_job["config_path"] == str(train["config_path"])
    assert train_job["log_path"] == str(train["log_path"])
    assert train_job["output_path"] == str(train["output_path"])
    assert train_job["process"]["pid"] == 4321
    assert train_job["child_jobs"][0]["job_id"] == "analyze_001"
    assert [item["iteration"] for item in train_job["finished_iterations"]] == [0, 1]

    assert analysis_job["config_path"] == str(analysis["config_path"])
    assert analysis_job["log_path"] == str(analysis["log_path"])
    assert analysis_job["output_path"] == str(analysis["result_dir"])
    assert analysis_job["parent_job"]["job_id"] == "train_001"
    assert analysis_job["progress_percent"] == pytest.approx(100.0)
    assert analysis_job["visualizations"]["manifest"]["counts"]["successes"] == 0


@pytest.mark.parametrize(
    ("text", "config", "status", "expected_stage", "expected_progress", "expected_epoch"),
    [
        (
            "\n".join(
                [
                    "[UNAGI] iteration 0: loading staged data",
                    "[UNAGI] iteration 0: training model",
                    "[TRAIN] iteration 0 epoch 1/1 loss=100.0",
                    "[UNAGI] iteration 0: running CPO",
                    "[UNAGI] iteration 0: updating stage attributes",
                    "[UNAGI] iteration 0: building temporal dynamics graph",
                    "[UNAGI] iteration 0: running iDREM",
                    "[UNAGI] iteration 0: updating gene weights",
                    "[UNAGI] iteration 0: building merged iteration dataset",
                    "Iteration 0 training is done!",
                    "[UNAGI] iteration 1: loading staged data",
                    "[UNAGI] iteration 1: training model",
                    "[TRAIN] iteration 1: initialize from previous iteration checkpoint",
                    "[TRAIN] iteration 1 epoch 2/5 loss=80.0",
                ]
            ),
            {"training": {"max_iter": 2}},
            "running",
            "training_model",
            63.5,
            2,
        ),
        (
            "\n".join(
                [
                    "[UNAGI-TRAIN] iteration=0 max_iter=3 stage=load_staged_data status=START",
                    "[UNAGI-TRAIN] iteration=0 max_iter=3 stage=load_staged_data status=DONE",
                    "[UNAGI-TRAIN] iteration=0 max_iter=3 stage=training_model status=START",
                    "[UNAGI-TRAIN] iteration=0 epoch=3/4 loss=12.0",
                ]
            ),
            {"training": {}},
            "running",
            "training_model",
            15.42,
            3,
        ),
        (
            "\n".join(
                [
                    "[UNAGI-TRAIN] iteration=0 max_iter=2 stage=load_staged_data status=START",
                    "[UNAGI-TRAIN] iteration=0 max_iter=2 stage=load_staged_data status=DONE",
                    "[UNAGI] iteration 0: training model",
                    "[UNAGI-TRAIN] iteration=0 epoch=1/2 loss=50.0",
                    "[UNAGI-TRAIN] iteration=0 max_iter=2 stage=training_model status=DONE",
                    "[UNAGI-TRAIN] iteration=0 max_iter=2 stage=cpo status=START",
                ]
            ),
            {"training": {"max_iter": 2}},
            "running",
            "cpo",
            30.0,
            None,
        ),
    ],
)
def test_parse_training_progress_supports_legacy_structured_and_mixed(
    tmp_path,
    text,
    config,
    status,
    expected_stage,
    expected_progress,
    expected_epoch,
):
    skill_root, repo_root, _skill_dir = create_monitor_workspace(tmp_path)
    service = JobMonitorService(skill_root=skill_root, repo_root=repo_root)

    progress = service._parse_training_progress(
        text,
        config,
        {"status": status, "started_at": "2026-03-06T10:00:00+00:00"},
    )

    assert progress["current_stage"] == expected_stage
    assert progress["progress_percent"] == pytest.approx(expected_progress, abs=0.01)
    assert progress["current_epoch"] == expected_epoch


def test_parse_analysis_progress_hierarchical_only_run_renormalizes_weights(tmp_path):
    skill_root, repo_root, _skill_dir = create_monitor_workspace(tmp_path)
    service = JobMonitorService(skill_root=skill_root, repo_root=repo_root)
    config = {
        "iteration": 1,
        "run_perturbation": False,
        "ignore_hcmarkers": False,
        "ignore_dynamic_markers": False,
    }
    text = "\n".join(
        [
            "[UNAGI-ANALYSIS] start analyse iteration=1, run_perturbation=False, centroid=True, "
            "perturb_change=0.5, direction_mode=one_sided",
            "[UNAGI-ANALYSIS] stage=hierarchical_markers status=START",
            "[UNAGI-ANALYSIS] stage=hierarchical_markers status=DONE",
            "[UNAGI-ANALYSIS] stage=pathway_overlap status=START",
            "[UNAGI-ANALYSIS] stage=pathway_overlap status=DONE",
            "[UNAGI-ANALYSIS] stage=pathway_ranking status=START",
            "[UNAGI-ANALYSIS] stage=pathway_ranking status=DONE",
            "[UNAGI-ANALYSIS] stage=copy_model_artifacts status=DONE",
            "[UNAGI-ANALYSIS] stage=progression_markers status=START",
            "[UNAGI-ANALYSIS] stage=progression_markers status=DONE",
            "[UNAGI-ANALYSIS] stage=save_outputs status=START",
            "[UNAGI-ANALYSIS] stage=save_outputs status=DONE",
            "[ANALYSIS] DONE",
        ]
    )

    progress = service._parse_analysis_progress(
        text,
        config,
        {"analysis_iteration": 1, "started_at": "2026-03-06T11:00:00+00:00"},
    )

    assert "perturbation" not in progress["stage_weights"]
    assert sum(progress["stage_weights"].values()) == pytest.approx(1.0)
    assert progress["progress_percent"] == pytest.approx(100.0)
    assert progress["current_stage"] is None


@pytest.mark.parametrize("direction_mode", ["one_sided", "two_sided"])
def test_parse_analysis_progress_tracks_perturbation_subprogress_and_modes(tmp_path, direction_mode):
    skill_root, repo_root, _skill_dir = create_monitor_workspace(tmp_path)
    service = JobMonitorService(skill_root=skill_root, repo_root=repo_root)
    config = {
        "iteration": 2,
        "run_perturbation": True,
        "ignore_hcmarkers": True,
        "ignore_dynamic_markers": True,
        "ignore_pathway_perturabtion": False,
        "ignore_drug_perturabtion": True,
        "centroid": direction_mode == "one_sided",
        "direction_mode": direction_mode,
    }
    text = "\n".join(
        [
            f"[UNAGI-ANALYSIS] start analyse iteration=2, run_perturbation=True, centroid={config['centroid']}, "
            f"perturb_change=0.5, direction_mode={direction_mode}",
            "[UNAGI-ANALYSIS] stage=pathway_overlap status=START",
            "[UNAGI-ANALYSIS] stage=pathway_overlap status=DONE",
            "[UNAGI-ANALYSIS] stage=pathway_ranking status=START",
            "[UNAGI-ANALYSIS] stage=pathway_ranking status=DONE",
            "[UNAGI-ANALYSIS] stage=copy_model_artifacts status=DONE",
            "[UNAGI-ANALYSIS] stage=perturbation status=START",
            "[UNAGI-ANALYSIS] task=pathway_perturbation status=START",
            "[UNAGI-PERTURB] mode=random_pathway_background status=START",
            "[UNAGI-PERTURB] mode=random_pathway_background finished=10/20 elapsed=00:01:00 eta=00:01:00",
        ]
    )

    progress = service._parse_analysis_progress(
        text,
        config,
        {"analysis_iteration": 2, "started_at": "2026-03-06T11:00:00+00:00"},
    )

    assert progress["enabled_tasks"] == ["pathway_perturbation"]
    assert progress["current_stage"] == "pathway_perturbation"
    assert progress["current_stage_label"] == "Pathway Perturbation"
    assert progress["progress_percent"] == pytest.approx(31.41, abs=0.01)
    assert progress["eta_seconds"] is not None


def test_parse_analysis_progress_tracks_generate_visualizations_stage(tmp_path, quiet_monitor):
    skill_root, repo_root, _skill_dir = create_monitor_workspace(tmp_path)
    service = JobMonitorService(skill_root=skill_root, repo_root=repo_root)
    config = {
        "iteration": 2,
        "run_perturbation": True,
        "generate_visualizations": True,
    }
    text = "\n".join(
        [
            "[UNAGI-ANALYSIS] stage=save_outputs status=DONE",
            "[UNAGI-ANALYSIS] stage=generate_visualizations status=START",
        ]
    )
    progress = service._parse_analysis_progress(text, config, {"analysis_iteration": 2})
    assert progress["current_stage"] == "generate_visualizations"
    assert progress["current_stage_label"] == "Generate visualizations"
    assert progress["progress_percent"] < 100.0


def test_parse_custom_perturbation_progress_and_parent_child_linking(tmp_path, monkeypatch, quiet_monitor):
    skill_root, repo_root, _skill_dir = create_monitor_workspace(tmp_path)
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
        status="running",
        log_text="[UNAGI-ANALYSIS] stage=hierarchical_markers status=START\n",
    )
    create_custom_perturbation_job(
        analysis["job_dir"],
        analysis["result_dir"],
        job_id="perturb_child",
        job_name="Gene perturbation",
        parent_job_id="analyze_parent",
        iteration=1,
        status="running",
        finished_at=None,
        log_text="\n".join(
            [
                "[UNAGI-CUSTOM-PERTURB] stage=load_context status=START",
                "[UNAGI-CUSTOM-PERTURB] stage=load_context status=DONE",
                "[UNAGI-CUSTOM-PERTURB] stage=resolve_targets status=START",
                "[UNAGI-CUSTOM-PERTURB] stage=resolve_targets status=DONE",
                "[UNAGI-CUSTOM-PERTURB] stage=background status=START",
            ]
        ),
    )

    service = JobMonitorService(skill_root=skill_root, repo_root=repo_root)
    jobs = service.list_jobs()
    assert {job["job_id"] for job in jobs} == {"train_parent", "analyze_parent", "perturb_child"}

    perturb_job = next(job for job in jobs if job["job_id"] == "perturb_child")
    assert perturb_job["parent_job"]["job_id"] == "analyze_parent"
    assert perturb_job["current_stage"] == "background"
    assert perturb_job["current_stage_label"] == "Build background"
    assert perturb_job["progress_percent"] == pytest.approx(20.0)

    analysis_job = next(job for job in jobs if job["job_id"] == "analyze_parent")
    assert analysis_job["child_jobs"][0]["job_id"] == "perturb_child"
