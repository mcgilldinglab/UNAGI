from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from .telemetry import collect_process_metrics, collect_system_metrics

ACTIVE_STATUSES = {"queued", "running", "retrying", "cancelling"}

TRAIN_STAGE_WEIGHTS = {
    "load_staged_data": 0.05,
    "training_model": 0.55,
    "cpo": 0.05,
    "update_stage_attributes": 0.10,
    "build_temporal_dynamics_graph": 0.08,
    "run_idrem": 0.10,
    "update_gene_weights": 0.04,
    "build_merged_iteration_dataset": 0.03,
}

TRAIN_STAGE_LABELS = {
    "load_staged_data": "Load staged data",
    "training_model": "Train model",
    "cpo": "Run CPO",
    "update_stage_attributes": "Update stage attributes",
    "build_temporal_dynamics_graph": "Build temporal dynamics graph",
    "run_idrem": "Run iDREM",
    "update_gene_weights": "Update gene weights",
    "build_merged_iteration_dataset": "Build merged iteration dataset",
}

ANALYSIS_STAGE_WEIGHTS = {
    "hierarchical_markers": 15.0,
    "pathway_overlap": 5.0,
    "pathway_ranking": 5.0,
    "copy_model_artifacts": 5.0,
    "progression_markers": 15.0,
    "perturbation": 45.0,
    "save_outputs": 10.0,
    "generate_visualizations": 10.0,
}

ANALYSIS_STAGE_LABELS = {
    "hierarchical_markers": "Hierarchical markers",
    "pathway_overlap": "Pathway overlap",
    "pathway_ranking": "Pathway ranking",
    "copy_model_artifacts": "Copy model artifacts",
    "progression_markers": "Progression markers",
    "perturbation": "Perturbation analysis",
    "prepare_drug_targets": "Prepare drug targets",
    "perturbation_compute": "Perturbation compute",
    "save_outputs": "Save outputs",
    "generate_visualizations": "Generate visualizations",
}

PERTURB_TASK_MODES = {
    "pathway_perturbation": ["random_pathway_background", "pathway"],
    "drug_perturbation": ["drug", "random_drug_background"],
    "single_gene_perturbation": ["single_gene", "random_background"],
    "two_gene_perturbation": ["two_genes", "random_background"],
    "two_drug_perturbation": ["two_drugs", "random_drug_background"],
}

ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
TRAIN_STRUCT_STAGE_RE = re.compile(
    r"\[UNAGI-TRAIN\]\s+iteration=(?P<iteration>\d+)\s+max_iter=(?P<max_iter>\S+)\s+stage=(?P<stage>[a-z_]+)\s+status=(?P<status>START|DONE)"
)
TRAIN_STRUCT_EPOCH_RE = re.compile(
    r"\[UNAGI-TRAIN\]\s+iteration=(?P<iteration>\d+)\s+epoch=(?P<epoch>\d+)/(?P<total>\d+)\s+loss=(?P<loss>[-+0-9.eE]+)"
)
TRAIN_LEGACY_STAGE_RE = re.compile(r"\[UNAGI\]\s+iteration\s+(?P<iteration>\d+):\s+(?P<stage>.+)")
TRAIN_LEGACY_EPOCH_RE = re.compile(
    r"\[TRAIN\]\s+iteration\s+(?P<iteration>\d+)\s+epoch\s+(?P<epoch>\d+)/(?P<total>\d+)\s+loss=(?P<loss>[-+0-9.eE]+)"
)
ANALYSIS_KV_RE = re.compile(r"([a-zA-Z_]+)=([^,\s]+)")
ANALYSIS_STAGE_RE = re.compile(r"\[UNAGI-ANALYSIS\]\s+(?P<body>.+)")
PERTURB_PROGRESS_RE = re.compile(
    r"\[UNAGI-PERTURB\]\s+mode=(?P<mode>[a-z_]+)\s+finished=(?P<finished>\d+)/(?P<total>\d+)"
)
PERTURB_STATUS_RE = re.compile(r"\[UNAGI-PERTURB\]\s+mode=(?P<mode>[a-z_]+)\s+status=(?P<status>START|DONE)")
ANALYSIS_DONE_RE = re.compile(r"\[ANALYSIS\]\s+DONE")
CUSTOM_PERTURB_STAGE_RE = re.compile(r"\[UNAGI-CUSTOM-PERTURB\]\s+stage=(?P<stage>[a-z_]+)\s+status=(?P<status>START|DONE)")
CUSTOM_PERTURB_PROGRESS_RE = re.compile(
    r"\[UNAGI-CUSTOM-PERTURB\]\s+stage=(?P<stage>[a-z_]+)\s+finished=(?P<finished>\d+)/(?P<total>\d+)"
)

CUSTOM_PERTURB_STAGE_WEIGHTS = {
    "load_context": 0.10,
    "resolve_targets": 0.10,
    "background": 0.20,
    "run": 0.30,
    "score": 0.20,
    "save": 0.10,
}

CUSTOM_PERTURB_STAGE_LABELS = {
    "load_context": "Load context",
    "resolve_targets": "Resolve targets",
    "background": "Build background",
    "run": "Run perturbation",
    "score": "Score perturbation",
    "save": "Save outputs",
}


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _format_seconds(seconds: float | None) -> str | None:
    if seconds is None or math.isnan(seconds) or seconds < 0:
        return None
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:02d}m {secs:02d}s"


def _cache_key(path: Path | None) -> tuple[str, int | None, int | None]:
    if path is None:
        return ("", None, None)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (str(path), None, None)
    return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _safe_read_json(path: Path) -> dict:
    return json.loads(_safe_read_text(path) or "{}")


def _safe_read_yaml(path: Path) -> dict:
    text = _safe_read_text(path)
    if not text.strip():
        return {}
    data = yaml.safe_load(text) or {}
    return data if isinstance(data, dict) else {}


def _safe_read_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _resolve_job_path(candidate: Any, fallback: Path) -> Path:
    if candidate:
        path = Path(str(candidate)).expanduser()
        if path.exists():
            return path.resolve()
    return fallback.resolve() if fallback.exists() else fallback


def _legacy_stage_key(label: str) -> tuple[str | None, bool]:
    text = label.strip().lower()
    if text == "loading staged data":
        return "load_staged_data", False
    if text == "training model":
        return "training_model", False
    if text == "running cpo":
        return "cpo", False
    if text.startswith("skip cpo"):
        return "cpo", True
    if text == "updating stage attributes":
        return "update_stage_attributes", False
    if text == "building temporal dynamics graph":
        return "build_temporal_dynamics_graph", False
    if text == "running idrem":
        return "run_idrem", False
    if text == "updating gene weights":
        return "update_gene_weights", False
    if text == "building merged iteration dataset":
        return "build_merged_iteration_dataset", False
    if text == "completed":
        return "__completed__", True
    return None, False


class JobMonitorService:
    def __init__(self, *, skill_root: Path, repo_root: Path, job_roots: Iterable[Path] | None = None):
        self.skill_root = Path(skill_root).resolve()
        self.repo_root = Path(repo_root).resolve()
        default_jobs_root = self.skill_root / "outputs" / "skill_jobs"
        roots = [default_jobs_root]
        if job_roots:
            roots.extend(Path(root).resolve() for root in job_roots)
        unique_roots: List[Path] = []
        seen: set[str] = set()
        for root in roots:
            key = str(root.resolve())
            if key in seen:
                continue
            seen.add(key)
            unique_roots.append(root)
        self.jobs_roots = unique_roots
        self._config_cache: Dict[tuple[str, int | None, int | None], dict] = {}
        self._log_cache: Dict[tuple[str, int | None, int | None, str], dict] = {}

    def list_jobs(self, *, status: str | None = None, kind: str | None = None) -> List[dict]:
        system_snapshot = collect_system_metrics(self.skill_root)
        gpu_pid_map = system_snapshot.get("gpu_pid_map", {})
        jobs = [self._normalize_job(job_file, gpu_pid_map=gpu_pid_map) for job_file in self._iter_job_files()]
        parent_map: Dict[str, List[dict]] = {}
        job_map = {job["job_id"]: job for job in jobs}
        for job in jobs:
            parent_id = job.get("parent_job_id")
            if parent_id:
                parent_map.setdefault(parent_id, []).append(job)
        for job in jobs:
            children = sorted(
                parent_map.get(job["job_id"], []),
                key=lambda item: (item.get("created_at") or "", item["job_id"]),
            )
            job["child_jobs"] = [self._child_summary(item) for item in children]
            if job.get("parent_job_id") and job["parent_job_id"] in job_map:
                job["parent_job"] = self._child_summary(job_map[job["parent_job_id"]])
            else:
                job["parent_job"] = None

        if status:
            jobs = [job for job in jobs if job.get("status") == status]
        if kind:
            jobs = [job for job in jobs if job.get("job_kind") == kind]
        return sorted(jobs, key=self._sort_key)

    def get_job(self, job_id: str) -> dict:
        for job in self.list_jobs():
            if job["job_id"] == job_id:
                return job
        raise FileNotFoundError(f"Unknown job_id: {job_id}")

    def get_log_tail(self, job_id: str, *, lines: int = 160) -> str:
        job = self.get_job(job_id)
        log_path = Path(job["log_path"])
        text = _safe_read_text(log_path)
        if not text:
            return ""
        return "\n".join(text.splitlines()[-max(1, int(lines)) :])

    def get_system_snapshot(self) -> dict:
        snapshot = collect_system_metrics(self.skill_root)
        gpu_pid_map = snapshot.pop("gpu_pid_map", {})
        jobs = [self._normalize_job(job_file, gpu_pid_map=gpu_pid_map) for job_file in self._iter_job_files()]
        processes = []
        for job in jobs:
            process = job.get("process") or {}
            pid = process.get("pid")
            if not pid:
                continue
            proc = collect_process_metrics(pid, gpu_pid_map)
            if proc:
                proc["job_id"] = job["job_id"]
                proc["job_name"] = job["job_name"]
                processes.append(proc)
        snapshot["processes"] = processes
        return snapshot

    def _iter_job_files(self) -> Iterable[Path]:
        job_files: List[Path] = []
        seen: set[str] = set()
        for jobs_root in self.jobs_roots:
            if not jobs_root.exists():
                continue
            for path in jobs_root.rglob("job.json"):
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                job_files.append(path)
        return sorted(job_files)

    def _sort_key(self, job: dict) -> tuple[int, float, str]:
        status_rank = 0 if job.get("status") in ACTIVE_STATUSES else 1
        updated = _parse_iso(job.get("updated_at"))
        updated_ts = updated.timestamp() if updated else 0.0
        return (status_rank, -updated_ts, job["job_id"])

    def _child_summary(self, job: dict) -> dict:
        return {
            "job_id": job["job_id"],
            "job_name": job["job_name"],
            "job_kind": job["job_kind"],
            "status": job["status"],
            "analysis_iteration": job.get("analysis_iteration"),
            "progress_percent": job["progress_percent"],
            "current_stage": job["current_stage"],
            "updated_at": job["updated_at"],
        }

    def _discover_finished_iterations(self, output_path: Path) -> list[dict]:
        iterations: list[dict] = []
        if not output_path.exists():
            return iterations
        for child in sorted(output_path.iterdir(), key=lambda item: item.name):
            if not child.is_dir() or not child.name.isdigit():
                continue
            dataset_path = child / "stagedata" / "dataset.h5ad"
            if not dataset_path.exists():
                continue
            iterations.append(
                {
                    "iteration": int(child.name),
                    "dataset_path": str(dataset_path),
                    "stage_dir": str(child),
                }
            )
        return iterations

    def _visualization_summary(self, output_path: Path) -> dict:
        visualizations_dir = output_path / "visualizations"
        manifest_path = visualizations_dir / "manifest.json"
        status_path = output_path / "visualization_status.json"
        manifest = _safe_read_manifest(manifest_path)
        return {
            "dir": str(visualizations_dir),
            "manifest_path": str(manifest_path),
            "exists": visualizations_dir.exists(),
            "manifest": manifest,
            "status": _safe_read_manifest(status_path),
        }

    def _normalize_job(self, job_file: Path, *, gpu_pid_map: Dict[int, dict] | None = None) -> dict:
        payload = _safe_read_json(job_file)
        job_kind = str(payload.get("job_kind", "train"))
        job_id = str(payload.get("job_id", job_file.parent.name))
        job_name = str(payload.get("job_name", job_id))

        if job_kind == "analysis":
            fallback_config = job_file.parent / "analysis_params.yaml"
            fallback_log = job_file.parent / "analysis.log"
            analysis_iteration = int(payload.get("analysis_iteration") or 0)
            fallback_output = job_file.parent.parent.parent / "analysis" / f"iter{analysis_iteration}" / job_id / "results"
        elif job_kind == "custom_perturbation":
            fallback_config = job_file.parent / "request.json"
            fallback_log = job_file.parent / "perturbation.log"
            analysis_iteration = int(payload.get("analysis_iteration") or 0)
            fallback_output = job_file.parent / "results"
        else:
            fallback_config = job_file.parent / "config.yaml"
            fallback_log = job_file.parent / "training.log"
            analysis_iteration = None
            fallback_output = job_file.parent / "input"

        config_path = _resolve_job_path(payload.get("config_path"), fallback_config)
        log_path = _resolve_job_path(payload.get("log_path"), fallback_log)
        output_path = _resolve_job_path(payload.get("analysis_output_dir"), fallback_output)
        config = self._read_config(config_path)
        progress = self._read_progress(log_path, config, job_kind, payload)

        status = str(payload.get("status", "unknown"))
        started_at = _parse_iso(payload.get("started_at"))
        finished_at = _parse_iso(payload.get("finished_at"))
        runtime_end = finished_at or _now_utc()
        duration_seconds = (runtime_end - started_at).total_seconds() if started_at else None

        child_pid = payload.get("child_pid")
        worker_pid = payload.get("worker_pid")
        process = collect_process_metrics(child_pid, gpu_pid_map) or collect_process_metrics(worker_pid, gpu_pid_map)

        artifact_paths = self._artifact_paths(job_kind, config_path, log_path, output_path)
        current_stage = progress.get("current_stage")
        normalized = {
            "job_id": job_id,
            "job_file_path": str(job_file.resolve()),
            "job_name": job_name,
            "job_kind": job_kind,
            "status": status,
            "repo_root": str(payload.get("repo_root") or self.repo_root),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
            "started_at": payload.get("started_at"),
            "finished_at": payload.get("finished_at"),
            "attempts": int(payload.get("attempts") or 0),
            "max_retries": int(payload.get("max_retries") or 0),
            "exit_code": payload.get("exit_code"),
            "parent_job_id": payload.get("parent_job_id"),
            "analysis_iteration": analysis_iteration,
            "resolved_execution_backend": payload.get("resolved_execution_backend"),
            "cpu_workers": payload.get("cpu_workers"),
            "config_path": str(config_path),
            "log_path": str(log_path),
            "output_path": str(output_path),
            "artifact_paths": artifact_paths,
            "current_iteration": progress.get("current_iteration"),
            "total_iterations": progress.get("total_iterations"),
            "current_epoch": progress.get("current_epoch"),
            "total_epochs": progress.get("total_epochs"),
            "last_loss": progress.get("last_loss"),
            "progress_percent": progress.get("progress_percent"),
            "progress_ratio": progress.get("progress_ratio"),
            "current_stage": current_stage,
            "current_stage_label": progress.get("current_stage_label"),
            "eta_seconds": progress.get("eta_seconds"),
            "eta_human": _format_seconds(progress.get("eta_seconds")),
            "runtime_seconds": duration_seconds,
            "runtime_human": _format_seconds(duration_seconds),
            "process": process,
            "monitor": progress,
            "child_jobs": [],
            "finished_iterations": self._discover_finished_iterations(output_path) if job_kind == "train" else [],
            "visualizations": self._visualization_summary(output_path) if job_kind == "analysis" else None,
        }
        if status == "succeeded":
            normalized["progress_percent"] = 100.0
            normalized["progress_ratio"] = 1.0
            normalized["eta_seconds"] = 0.0
            normalized["eta_human"] = _format_seconds(0.0)
            if not current_stage:
                normalized["current_stage"] = "completed"
                normalized["current_stage_label"] = "Completed"
        elif status in {"failed", "cancelled"} and not current_stage:
            normalized["current_stage"] = "stopped"
            normalized["current_stage_label"] = "Stopped"
        return normalized

    def _artifact_paths(self, job_kind: str, config_path: Path, log_path: Path, output_path: Path) -> dict:
        paths = {
            "config": str(config_path),
            "log": str(log_path),
            "output": str(output_path),
            "items": [],
        }
        candidates = [config_path, log_path, output_path]
        if job_kind == "train":
            candidates.extend(
                [
                    output_path / "model_save",
                    output_path.parent / "analysis",
                ]
            )
        elif job_kind == "custom_perturbation":
            candidates.extend(
                [
                    output_path / "summary.json",
                    output_path / "result_table.csv",
                    output_path / "ranked_bar.png",
                    output_path / "ranked_bar.pdf",
                ]
            )
        else:
            candidates.extend(
                [
                    output_path / "model_save",
                    output_path / "idrem",
                    output_path / "visualizations",
                    output_path / "visualizations" / "manifest.json",
                ]
            )
        paths["items"] = [str(path) for path in candidates if path.exists()]
        return paths

    def _read_config(self, config_path: Path) -> dict:
        key = _cache_key(config_path)
        cached = self._config_cache.get(key)
        if cached is not None:
            return cached
        payload = _safe_read_yaml(config_path)
        self._config_cache[key] = payload
        return payload

    def _read_progress(self, log_path: Path, config: dict, job_kind: str, payload: dict) -> dict:
        key = _cache_key(log_path) + (job_kind,)
        cached = self._log_cache.get(key)
        if cached is not None:
            return cached
        text = _safe_read_text(log_path)
        if job_kind == "analysis":
            progress = self._parse_analysis_progress(text, config, payload)
        elif job_kind == "custom_perturbation":
            progress = self._parse_custom_perturbation_progress(text, config, payload)
        else:
            progress = self._parse_training_progress(text, config, payload)
        self._log_cache[key] = progress
        return progress

    def _parse_custom_perturbation_progress(self, text: str, config: dict, payload: dict) -> dict:
        done_stages: set[str] = set()
        started_stages: set[str] = set()
        current_stage = None
        run_finished = None
        run_total = None
        eta_seconds = None

        for line in text.splitlines():
            match = CUSTOM_PERTURB_STAGE_RE.search(line)
            if match:
                stage = match.group("stage")
                status = match.group("status")
                if status == "START":
                    started_stages.add(stage)
                    current_stage = stage
                elif status == "DONE":
                    done_stages.add(stage)
                    if current_stage == stage:
                        current_stage = None
                continue

            match = CUSTOM_PERTURB_PROGRESS_RE.search(line)
            if match:
                current_stage = match.group("stage")
                run_finished = int(match.group("finished"))
                run_total = int(match.group("total"))

        progress_ratio = 0.0
        for stage, weight in CUSTOM_PERTURB_STAGE_WEIGHTS.items():
            if stage in done_stages:
                progress_ratio += weight

        if (
            current_stage == "run"
            and run_finished is not None
            and run_total
            and "run" not in done_stages
        ):
            stage_weight = CUSTOM_PERTURB_STAGE_WEIGHTS["run"]
            progress_ratio += stage_weight * min(1.0, run_finished / run_total)

        if current_stage is None:
            for stage in CUSTOM_PERTURB_STAGE_WEIGHTS:
                if stage in started_stages and stage not in done_stages:
                    current_stage = stage
                    break

        started_at = _parse_iso(payload.get("started_at"))
        if started_at and progress_ratio > 0 and progress_ratio < 1:
            elapsed = (_now_utc() - started_at).total_seconds()
            eta_seconds = elapsed * (1.0 - progress_ratio) / progress_ratio

        return {
            "current_iteration": payload.get("analysis_iteration"),
            "total_iterations": None,
            "current_epoch": None,
            "total_epochs": None,
            "last_loss": None,
            "current_stage": current_stage,
            "current_stage_label": CUSTOM_PERTURB_STAGE_LABELS.get(current_stage),
            "progress_ratio": min(1.0, progress_ratio),
            "progress_percent": round(min(1.0, progress_ratio) * 100.0, 2),
            "eta_seconds": eta_seconds,
            "done_stages": sorted(done_stages),
            "run_finished": run_finished,
            "run_total": run_total,
        }

    def _parse_training_progress(self, text: str, config: dict, payload: dict) -> dict:
        max_iter = int(((config.get("training") or {}).get("max_iter")) or 0)
        iterations: Dict[int, dict] = {}

        def ensure_iteration(idx: int) -> dict:
            return iterations.setdefault(
                idx,
                {
                    "done_stages": set(),
                    "current_stage": None,
                    "current_epoch": None,
                    "total_epochs": None,
                    "last_loss": None,
                    "completed": False,
                },
            )

        for line in text.splitlines():
            match = TRAIN_STRUCT_STAGE_RE.search(line)
            if match:
                iteration = int(match.group("iteration"))
                stage = match.group("stage")
                status = match.group("status")
                if max_iter <= 0:
                    value = match.group("max_iter")
                    if value.isdigit():
                        max_iter = int(value)
                entry = ensure_iteration(iteration)
                entry["current_stage"] = stage
                if status == "DONE":
                    entry["done_stages"].add(stage)
                    if stage == "build_merged_iteration_dataset":
                        entry["completed"] = True
                        entry["current_stage"] = None
                continue

            match = TRAIN_STRUCT_EPOCH_RE.search(line)
            if match:
                iteration = int(match.group("iteration"))
                entry = ensure_iteration(iteration)
                entry["current_stage"] = "training_model"
                entry["current_epoch"] = int(match.group("epoch"))
                entry["total_epochs"] = int(match.group("total"))
                entry["last_loss"] = float(match.group("loss"))
                continue

            match = TRAIN_LEGACY_STAGE_RE.search(line)
            if match:
                iteration = int(match.group("iteration"))
                stage_key, immediate_done = _legacy_stage_key(match.group("stage"))
                if stage_key is None:
                    continue
                entry = ensure_iteration(iteration)
                previous = entry["current_stage"]
                if previous and previous in TRAIN_STAGE_WEIGHTS and previous != stage_key:
                    entry["done_stages"].add(previous)
                if stage_key == "__completed__":
                    entry["done_stages"].update(TRAIN_STAGE_WEIGHTS.keys())
                    entry["completed"] = True
                    entry["current_stage"] = None
                else:
                    entry["current_stage"] = stage_key
                    if immediate_done:
                        entry["done_stages"].add(stage_key)
                continue

            match = TRAIN_LEGACY_EPOCH_RE.search(line)
            if match:
                iteration = int(match.group("iteration"))
                entry = ensure_iteration(iteration)
                entry["current_stage"] = "training_model"
                entry["current_epoch"] = int(match.group("epoch"))
                entry["total_epochs"] = int(match.group("total"))
                entry["last_loss"] = float(match.group("loss"))
                continue

            if line.startswith("Iteration ") and "training is done!" in line:
                try:
                    iteration = int(line.split()[1])
                except (IndexError, ValueError):
                    continue
                entry = ensure_iteration(iteration)
                entry["done_stages"].update(TRAIN_STAGE_WEIGHTS.keys())
                entry["completed"] = True
                entry["current_stage"] = None

        if max_iter <= 0 and iterations:
            max_iter = max(iterations) + 1

        completed_iterations = [idx for idx, value in iterations.items() if value["completed"]]
        current_iteration = None
        current_entry = None
        if payload.get("status") == "succeeded" and max_iter > 0:
            current_iteration = max_iter - 1
            current_entry = iterations.get(current_iteration)
        elif iterations:
            open_iterations = sorted(idx for idx, value in iterations.items() if not value["completed"])
            if open_iterations:
                current_iteration = open_iterations[-1]
                current_entry = iterations[current_iteration]
            else:
                current_iteration = max(iterations)
                current_entry = iterations[current_iteration]

        progress_ratio = 0.0
        current_stage = None
        current_epoch = None
        total_epochs = None
        last_loss = None
        eta_seconds = None

        if max_iter > 0:
            progress_ratio += len(completed_iterations) / max_iter
            if current_iteration is not None and current_entry and not current_entry["completed"]:
                iter_fraction = 0.0
                for stage, weight in TRAIN_STAGE_WEIGHTS.items():
                    if stage in current_entry["done_stages"]:
                        iter_fraction += weight
                stage = current_entry.get("current_stage")
                if stage == "training_model":
                    epoch = current_entry.get("current_epoch")
                    epochs = current_entry.get("total_epochs")
                    if epoch and epochs:
                        iter_fraction += TRAIN_STAGE_WEIGHTS["training_model"] * (epoch / epochs)
                current_stage = stage
                current_epoch = current_entry.get("current_epoch")
                total_epochs = current_entry.get("total_epochs")
                last_loss = current_entry.get("last_loss")
                progress_ratio = min(1.0, progress_ratio + (iter_fraction / max_iter))
                if current_stage == "training_model" and current_epoch and total_epochs and current_epoch > 0:
                    started_at = _parse_iso(payload.get("started_at"))
                    if started_at:
                        elapsed = (_now_utc() - started_at).total_seconds()
                        completed_fraction = progress_ratio
                        if completed_fraction > 0:
                            eta_seconds = elapsed * (1.0 - completed_fraction) / completed_fraction
                else:
                    current_epoch = None
                    total_epochs = None
            elif current_entry:
                last_loss = current_entry.get("last_loss")
        progress_percent = round(progress_ratio * 100.0, 2)
        current_stage_label = TRAIN_STAGE_LABELS.get(current_stage) if current_stage else None
        return {
            "current_iteration": current_iteration,
            "total_iterations": max_iter or None,
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "last_loss": last_loss,
            "current_stage": current_stage,
            "current_stage_label": current_stage_label,
            "progress_ratio": progress_ratio,
            "progress_percent": progress_percent,
            "eta_seconds": eta_seconds,
            "iterations": {
                str(idx): {
                    "completed": bool(value["completed"]),
                    "done_stages": sorted(value["done_stages"]),
                    "current_stage": value["current_stage"],
                    "current_epoch": value["current_epoch"],
                    "total_epochs": value["total_epochs"],
                    "last_loss": value["last_loss"],
                }
                for idx, value in sorted(iterations.items())
            },
        }

    def _analysis_weights(self, config: dict) -> tuple[dict, List[str]]:
        run_perturbation = bool(config.get("run_perturbation", True))
        include_hc = not bool(config.get("ignore_hcmarkers", False))
        include_dynamic = not bool(config.get("ignore_dynamic_markers", False))
        include_pathway = run_perturbation and not bool(config.get("ignore_pathway_perturabtion", False))
        include_single_gene = run_perturbation and (
            bool(config.get("run_single_gene_perturbation", False))
            or (
                bool(config.get("run_two_gene_perturbation", False))
                and bool(config.get("shortlist_two_gene_candidates", False))
            )
        )
        include_two_gene = run_perturbation and bool(config.get("run_two_gene_perturbation", False))
        include_drug = run_perturbation and (
            not bool(config.get("ignore_drug_perturabtion", False))
            or (
                bool(config.get("run_two_drug_perturbation", False))
                and bool(config.get("shortlist_two_drug_candidates", False))
            )
        )
        include_two_drug = run_perturbation and bool(config.get("run_two_drug_perturbation", False))

        enabled = dict(ANALYSIS_STAGE_WEIGHTS)
        include_visualizations = bool(config.get("generate_visualizations", True))
        if not include_hc:
            enabled.pop("hierarchical_markers", None)
        if not include_dynamic:
            enabled.pop("progression_markers", None)
        if not run_perturbation or (not include_pathway and not include_drug and not include_single_gene and not include_two_gene and not include_two_drug):
            enabled.pop("perturbation", None)
        if not include_visualizations:
            enabled.pop("generate_visualizations", None)

        enabled_tasks = []
        if include_pathway:
            enabled_tasks.append("pathway_perturbation")
        if include_drug:
            enabled_tasks.append("drug_perturbation")
        if include_single_gene:
            enabled_tasks.append("single_gene_perturbation")
        if include_two_gene:
            enabled_tasks.append("two_gene_perturbation")
        if include_two_drug:
            enabled_tasks.append("two_drug_perturbation")

        total = sum(enabled.values()) or 1.0
        normalized = {stage: weight / total for stage, weight in enabled.items()}
        return normalized, enabled_tasks

    def _task_progress(self, task_name: str, task_state: dict, enabled_modes: List[str]) -> float:
        if task_state.get("status") == "DONE":
            return 1.0
        if task_state.get("status") == "SKIP":
            return 1.0
        mode_progress = 0.0
        total_modes = len(enabled_modes) or 1
        completed_modes = 0
        partial_mode = 0.0
        for mode in enabled_modes:
            mode_state = task_state["modes"].get(mode, {})
            if mode_state.get("status") == "DONE":
                completed_modes += 1
                continue
            finished = mode_state.get("finished")
            total = mode_state.get("total")
            if finished is not None and total:
                partial_mode = max(partial_mode, min(1.0, finished / total))
        mode_progress = (completed_modes + partial_mode) / total_modes
        return min(0.9, mode_progress * 0.9)

    def _parse_analysis_progress(self, text: str, config: dict, payload: dict) -> dict:
        weights, enabled_tasks = self._analysis_weights(config)
        done_stages = set()
        started_stages = set()
        task_state = {
            task: {"status": None, "modes": {}}
            for task in PERTURB_TASK_MODES
        }
        current_stage = None
        current_stage_label = None
        eta_seconds = None
        perturb_eta_seconds = None
        analysis_iteration = int(config.get("iteration") or payload.get("analysis_iteration") or 0) or None

        for line in text.splitlines():
            match = ANALYSIS_STAGE_RE.search(line)
            if match:
                body = match.group("body")
                pairs = {key: value for key, value in ANALYSIS_KV_RE.findall(body)}
                stage_name = pairs.get("stage")
                status = pairs.get("status")
                task_name = pairs.get("task")
                if stage_name and status:
                    if status == "START":
                        started_stages.add(stage_name)
                        current_stage = stage_name
                    elif status == "DONE":
                        done_stages.add(stage_name)
                        if current_stage == stage_name:
                            current_stage = None
                    elif status == "SKIP":
                        done_stages.add(stage_name)
                    continue
                if task_name and status:
                    state = task_state.setdefault(task_name, {"status": None, "modes": {}})
                    state["status"] = status
                    if status == "START":
                        current_stage = task_name
                    elif status == "DONE" and current_stage == task_name:
                        current_stage = None
                    continue

            match = PERTURB_STATUS_RE.search(line)
            if match:
                mode = match.group("mode")
                status = match.group("status")
                task_name = "pathway_perturbation" if "pathway" in mode else "drug_perturbation"
                state = task_state.setdefault(task_name, {"status": None, "modes": {}})
                mode_state = state["modes"].setdefault(mode, {})
                mode_state["status"] = status
                continue

            match = PERTURB_PROGRESS_RE.search(line)
            if match:
                mode = match.group("mode")
                task_name = "pathway_perturbation" if "pathway" in mode else "drug_perturbation"
                state = task_state.setdefault(task_name, {"status": None, "modes": {}})
                mode_state = state["modes"].setdefault(mode, {})
                mode_state["finished"] = int(match.group("finished"))
                mode_state["total"] = int(match.group("total"))
                current_stage = task_name
                continue

            if ANALYSIS_DONE_RE.search(line):
                done_stages.update(weights.keys())
                current_stage = None

        progress_ratio = 0.0
        for stage_name, weight in weights.items():
            if stage_name in done_stages:
                progress_ratio += weight

        if "perturbation" in weights and "perturbation" not in done_stages:
            if enabled_tasks:
                per_task_weight = weights["perturbation"] / len(enabled_tasks)
                task_progress_total = 0.0
                for task_name in enabled_tasks:
                    task_progress_total += self._task_progress(
                        task_name,
                        task_state.get(task_name, {"status": None, "modes": {}}),
                        PERTURB_TASK_MODES.get(task_name, []),
                    )
                progress_ratio += per_task_weight * task_progress_total
                for task_name in enabled_tasks:
                    for mode in task_state.get(task_name, {}).get("modes", {}).values():
                        finished = mode.get("finished")
                        total = mode.get("total")
                        if finished and total and finished > 0:
                            started_at = _parse_iso(payload.get("started_at"))
                            if started_at:
                                elapsed = (_now_utc() - started_at).total_seconds()
                                completed_fraction = progress_ratio
                                if completed_fraction > 0:
                                    perturb_eta_seconds = elapsed * (1.0 - completed_fraction) / completed_fraction
                                break
                    if perturb_eta_seconds is not None:
                        break

        if current_stage is None:
            for stage_name in weights:
                if stage_name not in done_stages and stage_name in started_stages:
                    current_stage = stage_name
                    break

        if current_stage in ANALYSIS_STAGE_LABELS:
            current_stage_label = ANALYSIS_STAGE_LABELS[current_stage]
        elif current_stage in {"pathway_perturbation", "drug_perturbation"}:
            current_stage_label = current_stage.replace("_", " ").title()

        eta_seconds = perturb_eta_seconds
        return {
            "current_iteration": analysis_iteration,
            "total_iterations": None,
            "current_epoch": None,
            "total_epochs": None,
            "last_loss": None,
            "current_stage": current_stage,
            "current_stage_label": current_stage_label,
            "progress_ratio": min(1.0, progress_ratio),
            "progress_percent": round(min(1.0, progress_ratio) * 100.0, 2),
            "eta_seconds": eta_seconds,
            "stage_weights": weights,
            "done_stages": sorted(done_stages),
            "enabled_tasks": enabled_tasks,
        }
