from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import yaml

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


def test_create_train_and_analysis_jobs_from_payload_and_manage_lifecycle(tmp_path, monkeypatch):
    skill_root, repo_root, _ = create_monitor_workspace(tmp_path)
    monkeypatch.setenv("UNAGI_SKILL_ROOT", str(skill_root))
    monkeypatch.setattr(job_backend, "_launch_worker", lambda repo_root, jf: (4242, ["python", "worker"]))

    dataset_path = repo_root / "demo.h5ad"
    dataset_path.write_text("stub", encoding="utf-8")

    training_payload = {
        "data": {
            "path": str(dataset_path),
            "stage_key": "stage",
            "total_stage": 2,
            "label_key": "name.simple",
        },
        "training": {
            "task": "train",
            "dist": "zinb",
            "GPU": False,
            "epoch_initial": 1,
            "epoch_iter": 2,
            "max_iter": 2,
            "BATCHSIZE": 16,
            "adversarial": True,
            "GCN": True,
        },
        "run": {"CPO": True, "connect_edges_cutoff": 0.05},
    }

    train_job = job_backend.create_job_from_config_payload(
        repo_root=repo_root,
        config_payload=training_payload,
        source_config_path="configs/unagi_workflow.example.run.yaml",
        job_name="Payload Training",
    )

    train_job_dir = skill_root / "outputs" / "skill_jobs" / train_job["job_id"]
    written_cfg = yaml.safe_load((train_job_dir / "config.yaml").read_text(encoding="utf-8"))
    assert written_cfg["data"]["path"] == str(train_job_dir / "input" / "demo.h5ad")
    assert train_job["worker_pid"] == 4242

    created_train = create_train_job(
        skill_root,
        job_id="train_existing",
        job_name="Existing Training",
        status="succeeded",
        log_text="Iteration 1 training is done!\n",
        finished_iterations=[1],
    )

    analysis_job = job_backend.create_analysis_job_from_params(
        repo_root=repo_root,
        training_job_id="train_existing",
        iteration=1,
        params_payload={"run_perturbation": False, "direction_mode": "one_sided"},
        analysis_config_source="configs/unagi_analysis_template.yaml",
        job_name="Payload Analysis",
    )
    assert analysis_job["analysis_iteration"] == 1
    assert analysis_job["worker_pid"] == 4242

    listed = job_backend.list_jobs_payloads(repo_root=repo_root)
    assert {row["job_id"] for row in listed} >= {train_job["job_id"], "train_existing", analysis_job["job_id"]}

    cancel_result = job_backend.cancel_job(repo_root=repo_root, job_id=train_job["job_id"])
    assert cancel_result["status"] == "cancel_requested"

    job_file = train_job_dir / "job.json"
    payload = json.loads(job_file.read_text(encoding="utf-8"))
    payload["status"] = "failed"
    payload["worker_pid"] = 0
    job_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    resume_result = job_backend.resume_job(repo_root=repo_root, job_id=train_job["job_id"])
    assert resume_result["status"] == "resubmitted"
    assert resume_result["worker_pid"] == 4242

    log_path = train_job_dir / "training.log"
    log_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
    assert job_backend.tail_job_log(repo_root=repo_root, job_id=train_job["job_id"], lines=2) == "line2\nline3"


def test_delete_job_supports_analysis_cleanup_partial_missing_paths_and_train_cascade(tmp_path, monkeypatch):
    skill_root, repo_root, _ = create_monitor_workspace(tmp_path)
    monkeypatch.setenv("UNAGI_SKILL_ROOT", str(skill_root))

    running_train = create_train_job(
        skill_root,
        job_id="train_running",
        job_name="Running Training",
        status="running",
        log_text="still running\n",
    )
    try:
        job_backend.delete_job(repo_root=repo_root, job_id="train_running")
    except ValueError as exc:
        assert "terminal state" in str(exc)
    else:
        raise AssertionError("Expected active training job deletion to be blocked")
    assert running_train["job_dir"].exists()

    terminal_train = create_train_job(
        skill_root,
        job_id="train_terminal",
        job_name="Terminal Training",
        status="succeeded",
        log_text="done\n",
        finished_iterations=[1],
    )
    partial_analysis = create_analysis_job(
        terminal_train["job_dir"],
        job_id="analyze_partial",
        job_name="Partial Analysis",
        parent_job_id="train_terminal",
        iteration=1,
        status="failed",
        log_text="failed\n",
    )
    terminal_analysis = create_analysis_job(
        terminal_train["job_dir"],
        job_id="analyze_terminal",
        job_name="Terminal Analysis",
        parent_job_id="train_terminal",
        iteration=1,
        status="succeeded",
        log_text="done\n",
    )

    import shutil
    shutil.rmtree(partial_analysis["result_dir"].parent)

    deleted_analysis = job_backend.delete_job(repo_root=repo_root, job_id="analyze_partial")
    assert deleted_analysis["status"] == "deleted"
    assert any("already missing" in warning for warning in deleted_analysis["warnings"])
    assert not partial_analysis["job_dir"].exists()
    assert not (terminal_train["job_dir"] / "analysis_jobs" / "analyze_partial").exists()

    deleted_train = job_backend.delete_job(repo_root=repo_root, job_id="train_terminal")
    assert deleted_train["status"] == "deleted"
    assert str(terminal_train["job_dir"]) in deleted_train["deleted_paths"]
    assert not terminal_train["job_dir"].exists()
    assert not terminal_analysis["job_dir"].exists()


def test_delete_jobs_bulk_reports_deleted_blocked_and_failed(tmp_path, monkeypatch):
    skill_root, repo_root, _ = create_monitor_workspace(tmp_path)
    monkeypatch.setenv("UNAGI_SKILL_ROOT", str(skill_root))

    deletable = create_train_job(
        skill_root,
        job_id="train_delete_me",
        job_name="Delete Me",
        status="failed",
        log_text="failed\n",
    )
    blocked = create_train_job(
        skill_root,
        job_id="train_blocked",
        job_name="Blocked",
        status="queued",
        log_text="queued\n",
    )

    result = job_backend.delete_jobs(
        repo_root=repo_root,
        job_ids=["train_delete_me", "train_blocked", "missing_job"],
    )
    assert result["status"] == "ok"
    assert [item["job_id"] for item in result["deleted"]] == ["train_delete_me"]
    assert [item["job_id"] for item in result["blocked"]] == ["train_blocked"]
    assert [item["job_id"] for item in result["failed"]] == ["missing_job"]
    assert not deletable["job_dir"].exists()
    assert blocked["job_dir"].exists()


def test_cancel_and_resume_support_explicit_job_files_for_scanned_jobs(tmp_path, monkeypatch):
    skill_root, repo_root, _ = create_monitor_workspace(tmp_path)
    monkeypatch.setenv("UNAGI_SKILL_ROOT", str(skill_root))
    monkeypatch.setattr(job_backend, "_launch_worker", lambda repo_root, jf: (4242, ["python", "worker"]))

    external_root = tmp_path / "external_workspace"
    train_job = create_train_job(
        external_root,
        job_id="train_external_resume",
        job_name="External Resume Job",
        status="queued",
        log_text="queued\n",
    )

    cancel_result = job_backend.cancel_job_from_file(job_file_path=train_job["job_dir"] / "job.json")
    assert cancel_result["status"] == "cancel_requested"

    payload = json.loads((train_job["job_dir"] / "job.json").read_text(encoding="utf-8"))
    payload["status"] = "failed"
    payload["worker_pid"] = 0
    (train_job["job_dir"] / "job.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    resume_result = job_backend.resume_job_from_file(
        job_file_path=train_job["job_dir"] / "job.json",
        repo_root=repo_root,
    )
    assert resume_result["status"] == "resubmitted"
    assert resume_result["worker_pid"] == 4242


def test_create_and_manage_custom_perturbation_jobs(tmp_path, monkeypatch):
    skill_root, repo_root, _ = create_monitor_workspace(tmp_path)
    monkeypatch.setenv("UNAGI_SKILL_ROOT", str(skill_root))
    monkeypatch.setattr(job_backend, "_launch_worker", lambda repo_root, jf: (4242, ["python", "worker"]))
    monkeypatch.setattr(
        job_backend,
        "_ensure_custom_perturbation_daemon",
        lambda **kwargs: (5252, ["python", "perturb-daemon", "--queue-key", kwargs["queue_key"]]),
    )

    train_job = create_train_job(
        skill_root,
        job_id="train_custom_parent",
        job_name="Training Parent",
        status="succeeded",
        finished_iterations=[1],
        log_text="Iteration 1 training is done!\n",
    )
    analysis_job = create_analysis_job(
        train_job["job_dir"],
        job_id="analyze_custom_parent",
        job_name="Analysis Parent",
        parent_job_id="train_custom_parent",
        iteration=1,
        status="succeeded",
        stale_paths=True,
        with_visualization_manifest=False,
        log_text="[ANALYSIS] DONE\n",
    )
    write_analysis_dataset(analysis_job["result_dir"])

    request_payload = {
        "mode": "gene",
        "scope_kind": "track",
        "track_key": "0-2-4",
        "perturb_change": 0.5,
        "direction_mode": "two_sided",
        "execution_mode": "auto",
        "gene_text": "",
        "signed_gene_text": "gene0:+, gene1:-",
        "target_name": "",
    }

    perturb_job = job_backend.create_custom_perturbation_job_from_payload(
        repo_root=repo_root,
        analysis_job_id="analyze_custom_parent",
        request_payload=request_payload,
    )

    perturb_job_dir = analysis_job["job_dir"] / "perturbation_jobs" / perturb_job["job_id"]
    result_dir = analysis_job["result_dir"] / "custom_perturbations" / perturb_job["job_id"]
    request_json = json.loads((perturb_job_dir / "request.json").read_text(encoding="utf-8"))

    assert perturb_job["job_kind"] == "custom_perturbation"
    assert perturb_job["job_name"] == "GENE0:+, GENE1:-"
    assert perturb_job["worker_pid"] == 5252
    assert perturb_job["execution_mode"] in {"cell_exact", "centroid"}
    assert perturb_job["resolved_execution_backend"] in {"gpu_exact", "cpu_exact_parallel", "gpu_centroid", "cpu_centroid"}
    assert int(perturb_job["cpu_workers"]) >= 1
    assert Path(request_json["analysis_dir"]) == analysis_job["result_dir"]


def test_create_custom_perturbation_job_from_explicit_analysis_job_file(tmp_path, monkeypatch):
    skill_root, repo_root, _ = create_monitor_workspace(tmp_path)
    monkeypatch.setenv("UNAGI_SKILL_ROOT", str(skill_root))
    daemon_calls: list[dict] = []
    monkeypatch.setattr(
        job_backend,
        "_ensure_custom_perturbation_daemon",
        lambda **kwargs: daemon_calls.append(kwargs) or (5252, ["python", "perturb-daemon", "--queue-key", kwargs["queue_key"]]),
    )

    external_root = tmp_path / "external_workspace"
    train_job = create_train_job(
        external_root,
        job_id="train_external_parent",
        job_name="External Training Parent",
        status="succeeded",
        finished_iterations=[1],
        log_text="Iteration 1 training is done!\n",
    )
    analysis_job = create_analysis_job(
        train_job["job_dir"],
        job_id="analyze_external_parent",
        job_name="External Analysis Parent",
        parent_job_id="train_external_parent",
        iteration=1,
        status="succeeded",
        stale_paths=True,
        with_visualization_manifest=False,
        log_text="[ANALYSIS] DONE\n",
    )
    write_analysis_dataset(analysis_job["result_dir"])

    perturb_job = job_backend.create_custom_perturbation_job_from_file(
        repo_root=repo_root,
        analysis_job_file_path=analysis_job["job_json"],
        analysis_job_id="analyze_external_parent",
        request_payload={
            "mode": "gene",
            "scope_kind": "overall",
            "perturb_change": 0.5,
            "direction_mode": "two_sided",
            "execution_mode": "auto",
            "signed_gene_text": "gene0:+",
        },
    )

    perturb_job_dir = analysis_job["job_dir"] / "perturbation_jobs" / perturb_job["job_id"]
    result_dir = analysis_job["result_dir"] / "custom_perturbations" / perturb_job["job_id"]
    request_json = json.loads((perturb_job_dir / "request.json").read_text(encoding="utf-8"))

    assert perturb_job["parent_job_id"] == "analyze_external_parent"
    assert perturb_job_dir.exists()
    assert result_dir.exists()
    assert request_json["analysis_job_id"] == "analyze_external_parent"
    assert Path(request_json["analysis_dir"]) == analysis_job["result_dir"]
    assert request_json["target_spec"]["signed_genes"] == ["GENE0:+"]
    assert result_dir.exists()
    assert daemon_calls[0]["watched_job_root"] == external_root / "outputs" / "skill_jobs"

    payload = json.loads((perturb_job_dir / "job.json").read_text(encoding="utf-8"))
    assert payload["worker_pid"] == 5252
    payload["status"] = "succeeded"
    (perturb_job_dir / "job.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (result_dir / "summary.json").write_text("{}", encoding="utf-8")

    deleted = job_backend.delete_job_from_file(job_file_path=perturb_job_dir / "job.json")
    assert deleted["status"] == "deleted"
    assert not perturb_job_dir.exists()
    assert not result_dir.exists()


def test_duplicate_active_custom_perturbation_submit_reuses_existing_job(tmp_path, monkeypatch):
    skill_root, repo_root, _ = create_monitor_workspace(tmp_path)
    monkeypatch.setenv("UNAGI_SKILL_ROOT", str(skill_root))
    monkeypatch.setattr(
        job_backend,
        "_ensure_custom_perturbation_daemon",
        lambda **kwargs: (5252, ["python", "perturb-daemon", "--queue-key", kwargs["queue_key"]]),
    )

    train_job = create_train_job(
        skill_root,
        job_id="train_dup_parent",
        job_name="Training Parent",
        status="succeeded",
        finished_iterations=[1],
        log_text="Iteration 1 training is done!\n",
    )
    analysis_job = create_analysis_job(
        train_job["job_dir"],
        job_id="analyze_dup_parent",
        job_name="Analysis Parent",
        parent_job_id="train_dup_parent",
        iteration=1,
        status="succeeded",
        stale_paths=True,
        with_visualization_manifest=False,
        log_text="[ANALYSIS] DONE\n",
    )
    write_analysis_dataset(analysis_job["result_dir"])

    request_payload = {
        "mode": "pathway",
        "scope_kind": "overall",
        "perturb_change": 0.5,
        "direction_mode": "two_sided",
        "execution_mode": "auto",
        "target_name": "TGFBETA PATHWAY",
        "gene_text": "gene0, gene1",
        "signed_gene_text": "",
    }
    first = job_backend.create_custom_perturbation_job_from_file(
        repo_root=repo_root,
        analysis_job_file_path=analysis_job["job_json"],
        analysis_job_id="analyze_dup_parent",
        request_payload=request_payload,
    )
    second = job_backend.create_custom_perturbation_job_from_file(
        repo_root=repo_root,
        analysis_job_file_path=analysis_job["job_json"],
        analysis_job_id="analyze_dup_parent",
        request_payload=request_payload,
    )

    assert second["job_id"] == first["job_id"]
    assert len(list((analysis_job["job_dir"] / "perturbation_jobs").glob("*/job.json"))) == 1


def test_default_custom_perturbation_job_names():
    assert job_backend._default_custom_perturbation_job_name(
        {"mode": "gene", "target_spec": {"signed_genes": ["SFTPC:+", "COL1A1:-"]}},
        "fallback",
    ) == "SFTPC:+, COL1A1:-"
    assert job_backend._default_custom_perturbation_job_name(
        {"mode": "pathway", "target_spec": {"name": "EMT pathway", "genes": ["VIM"]}},
        "fallback",
    ) == "EMT pathway"
    assert job_backend._default_custom_perturbation_job_name(
        {"mode": "drug", "target_spec": {"name": "nimodipine", "signed_genes": ["CACNA1C:+"]}},
        "fallback",
    ) == "nimodipine"


def test_ensure_custom_perturbation_daemon_restarts_on_code_change(tmp_path, monkeypatch):
    skill_root, repo_root, _ = create_monitor_workspace(tmp_path)
    monkeypatch.setenv("UNAGI_SKILL_ROOT", str(skill_root))

    train_job = create_train_job(
        skill_root,
        job_id="train_restart_parent",
        job_name="Training Parent",
        status="succeeded",
        finished_iterations=[1],
        log_text="Iteration 1 training is done!\n",
    )
    analysis_job = create_analysis_job(
        train_job["job_dir"],
        job_id="analyze_restart_parent",
        job_name="Analysis Parent",
        parent_job_id="train_restart_parent",
        iteration=1,
        status="succeeded",
        stale_paths=False,
        with_visualization_manifest=False,
        log_text="[ANALYSIS] DONE\n",
    )
    write_analysis_dataset(analysis_job["result_dir"])
    custom_job = create_custom_perturbation_job(
        train_job["job_dir"],
        analysis_job["result_dir"],
        job_id="perturb_running",
        job_name="Running custom perturbation",
        parent_job_id="analyze_restart_parent",
        iteration=1,
        status="running",
        finished_at=None,
        log_text="[UNAGI-CUSTOM-PERTURB] stage=run status=START\n",
    )
    payload = json.loads(custom_job["job_json"].read_text(encoding="utf-8"))
    payload["worker_pid"] = 123
    payload["child_pid"] = 123
    custom_job["job_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    state_path = skill_root / "outputs" / "perturbation_workers" / "gpu_0.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "queue_key": "gpu:0",
                "pid": 123,
                "command": ["python", "perturb-daemon"],
                "code_fingerprint": "old",
                "updated_at": "2026-03-10T22:00:00+00:00",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    alive = {123}
    terminated: list[int] = []

    class _FakeProc:
        def __init__(self, pid: int) -> None:
            self.pid = pid

    monkeypatch.setattr(job_backend, "_custom_perturbation_daemon_fingerprint", lambda: "new")
    monkeypatch.setattr(job_backend, "_pid_alive", lambda pid: int(pid) in alive)

    def _fake_terminate(pid: int) -> None:
        terminated.append(int(pid))
        alive.discard(int(pid))

    monkeypatch.setattr(job_backend, "_terminate_process", _fake_terminate)
    monkeypatch.setattr(job_backend.subprocess, "Popen", lambda *args, **kwargs: _FakeProc(999))

    worker_pid, cmd = job_backend._ensure_custom_perturbation_daemon(repo_root=repo_root, queue_key="gpu:0")

    assert worker_pid == 999
    assert cmd[-2:] == ["--queue-key", "gpu:0"]
    assert terminated == [123]

    requeued = json.loads(custom_job["job_json"].read_text(encoding="utf-8"))
    assert requeued["status"] == "queued"
    assert requeued["worker_pid"] is None
    assert requeued["child_pid"] is None
    assert requeued["last_error"] == "Requeued after perturbation daemon restart."

    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert state_payload["pid"] == 999
    assert state_payload["code_fingerprint"] == "new"
