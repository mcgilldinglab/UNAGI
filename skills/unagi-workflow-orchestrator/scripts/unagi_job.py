#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

DEMO_CONFIG_REL = Path("configs/unagi_workflow.example.run.yaml")
ANALYSIS_TEMPLATE_REL = Path("configs/unagi_analysis_template.yaml")
JOB_ROOT_REL = Path("outputs/skill_jobs")
PERTURBATION_WORKER_ROOT_REL = Path("outputs/perturbation_workers")
ACTIVE_STATUSES = {"queued", "running", "retrying", "cancelling"}
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
SKILL_WORKSPACE_ENV = "UNAGI_SKILL_ROOT"


def _load_custom_perturbation_backend():
    repo_root = infer_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return importlib.import_module("UNAGI.monitoring.custom_perturbation")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def infer_repo_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3],
        Path.cwd().resolve(),
    ]
    for candidate in candidates:
        if (candidate / "UNAGI" / "cli.py").exists():
            return candidate
    return candidates[0]


def skill_workspace_root() -> Path:
    override = os.environ.get(SKILL_WORKSPACE_ENV)
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] if len(here.parents) > 3 else None,
        here.parents[1],
        Path.cwd().resolve(),
    ]
    for candidate in candidates:
        if candidate and (candidate / "skills" / "unagi-workflow-orchestrator" / "SKILL.md").exists():
            return candidate
    return here.parents[1]


def resolve_repo_root(repo_root: Optional[str]) -> Path:
    if repo_root:
        return Path(repo_root).expanduser().resolve()
    return infer_repo_root()


def job_root(repo_root: Path) -> Path:
    root = skill_workspace_root() / JOB_ROOT_REL
    root.mkdir(parents=True, exist_ok=True)
    return root


def perturbation_worker_root() -> Path:
    root = skill_workspace_root() / PERTURBATION_WORKER_ROOT_REL
    root.mkdir(parents=True, exist_ok=True)
    return root


def new_job_id(prefix: str = "train") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}"


def job_dir(repo_root: Path, job_id: str) -> Path:
    return job_root(repo_root) / job_id


def job_file(repo_root: Path, job_id: str) -> Path:
    return job_dir(repo_root, job_id) / "job.json"


def resolve_job_file(repo_root: Path, job_id: str) -> Path:
    direct = job_file(repo_root, job_id)
    if direct.exists():
        return direct

    matches = [path for path in job_root(repo_root).rglob("job.json") if path.parent.name == job_id]
    if not matches:
        raise FileNotFoundError(f"Unknown job_id: {job_id}")
    if len(matches) > 1:
        raise ValueError(f"Multiple job files found for job_id {job_id}: {[str(p) for p in matches]}")
    return matches[0]


def _resolve_existing_path(candidate: Any, fallback: Path) -> Path:
    if candidate:
        path = Path(str(candidate)).expanduser()
        if path.exists():
            return path.resolve()
    return fallback.resolve() if fallback.exists() else fallback


def _resolve_analysis_output_dir_for_job(job_file_path: Path, payload: Dict[str, Any]) -> Path:
    job_dir_path = job_file_path.parent.resolve()
    iteration = int(payload.get("analysis_iteration") or 0)
    analysis_job_id = str(payload.get("job_id") or job_dir_path.name)
    fallback = job_dir_path.parents[1] / "analysis" / f"iter{iteration}" / analysis_job_id / "results"
    return _resolve_existing_path(payload.get("analysis_output_dir"), fallback)


def _job_root_from_job_file(job_file_path: Path | str) -> Path:
    path = Path(job_file_path).expanduser().resolve()
    for parent in path.parents:
        if parent.name == JOB_ROOT_REL.name:
            return parent
    raise ValueError(f"Could not determine owning {JOB_ROOT_REL} root for job file: {path}")


def _normalize_job_roots(repo_root: Path, roots: Iterable[Path | str] | None = None) -> list[Path]:
    candidates: list[Path] = [job_root(repo_root)]
    for raw_root in roots or []:
        if not raw_root:
            continue
        candidates.append(Path(raw_root).expanduser().resolve())
    resolved: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(candidate)
    return resolved


def _default_custom_perturbation_job_name(normalized: Dict[str, Any], fallback_job_id: str) -> str:
    mode = str(normalized.get("mode") or "")
    target_spec = normalized.get("target_spec") or {}
    if mode == "gene":
        genes = [str(item) for item in (target_spec.get("signed_genes") or target_spec.get("genes") or []) if str(item).strip()]
        if genes:
            return ", ".join(genes)
    elif mode in {"pathway", "drug"}:
        name = str(target_spec.get("name") or "").strip()
        if name:
            return name
    return fallback_job_id


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def resolve_default_config(repo_root: Path, config_path: Optional[str]) -> Path:
    if config_path:
        return resolve_path(repo_root, config_path)
    demo = (repo_root / DEMO_CONFIG_REL).resolve()
    if not demo.exists():
        raise FileNotFoundError(f"Default demo config not found: {demo}")
    return demo


def resolve_default_analysis_config(repo_root: Path, config_path: Optional[str]) -> Optional[Path]:
    if config_path:
        cfg = resolve_path(repo_root, config_path)
        if not cfg.exists():
            raise FileNotFoundError(f"Analysis config not found: {cfg}")
        return cfg
    template = (repo_root / ANALYSIS_TEMPLATE_REL).resolve()
    if template.exists():
        return template
    return None


def load_training_template(repo_root: Path, config_path: Optional[str]) -> tuple[Dict[str, Any], Path]:
    resolved = resolve_default_config(repo_root, config_path)
    return read_yaml(resolved), resolved


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage_input_copy(src_path: Path, target_dir: Path) -> Path:
    src = src_path.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if src.is_file():
        dst = target_dir / src.name
        _safe_link_or_copy(src, dst)
        return dst

    if not src.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {src}")

    h5ads = sorted(src.glob("*.h5ad"))
    if not h5ads:
        raise FileNotFoundError(f"No .h5ad found in input directory: {src}")

    for file_path in h5ads:
        _safe_link_or_copy(file_path, target_dir / file_path.name)
    return target_dir


def _ensure_dict(root: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = root.get(key)
    if value is None:
        root[key] = {}
        return root[key]
    if not isinstance(value, dict):
        raise ValueError(f"Override path conflict: '{key}' is not a mapping")
    return value


def _set_dotted(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    if not all(keys):
        raise ValueError(f"Invalid dotted key: {dotted_key}")
    node = cfg
    for key in keys[:-1]:
        node = _ensure_dict(node, key)
    node[keys[-1]] = value


def _parse_override(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"Invalid override '{raw}'. Use dotted.key=value")
    key, raw_value = raw.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid override '{raw}'. Empty key")
    value = yaml.safe_load(raw_value)
    return key, value


def apply_set_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    merged = dict(cfg)
    for raw in overrides:
        key, value = _parse_override(raw)
        _set_dotted(merged, key, value)
    return merged


def _coerce_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"1", "true", "yes", "y", "on"}:
            return True
        if norm in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Invalid boolean for {key}: {value!r}")


def _resolve_optional_path(repo_root: Path, value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(resolve_path(repo_root, str(value)))


def _analysis_defaults(iteration: int, data_path: Path, target_dir: Path) -> Dict[str, Any]:
    return {
        "data_path": str(data_path),
        "target_dir": str(target_dir),
        "iteration": int(iteration),
        "random_background_sampling_times": 20,
        "run_perturbation": True,
        "generate_visualizations": True,
        "customized_pathway": None,
        "customized_drug": None,
        "cmap_dir": None,
        "perturb_change": 0.5,
        "direction_mode": "two_sided",
        "overall_perturbation_analysis": True,
        "perturbed_tracks": "all",
        "perturb_stage": None,
        "ignore_pathway_perturabtion": False,
        "ignore_drug_perturabtion": False,
        "centroid": False,
        "cpu_workers": None,
        "ignore_hcmarkers": False,
        "ignore_dynamic_markers": False,
        "dynamic_marker_strategy": "optimized",
        "dynamic_marker_cpu_workers": None,
        "dynamic_marker_force_serial": False,
        "dynamic_marker_verify_legacy": False,
    }


def _load_analysis_template(repo_root: Path, config_path: Optional[str]) -> tuple[Dict[str, Any], Optional[Path]]:
    resolved = resolve_default_analysis_config(repo_root, config_path)
    if resolved is None:
        return {}, None

    loaded = read_yaml(resolved)
    if "analysis" in loaded:
        nested = loaded.get("analysis")
        if not isinstance(nested, dict):
            raise ValueError(f"'analysis' section must be a mapping: {resolved}")
        return dict(nested), resolved
    return dict(loaded), resolved


def _normalize_analysis_params(
    raw_params: Dict[str, Any],
    *,
    repo_root: Path,
    iteration: int,
    data_path: Path,
    target_dir: Path,
    default_customized_drug_path: Path,
) -> Dict[str, Any]:
    params = dict(raw_params)

    params["data_path"] = str(data_path)
    params["target_dir"] = str(target_dir)
    params["iteration"] = int(iteration)
    params["random_background_sampling_times"] = int(params.get("random_background_sampling_times", 20))
    params["run_perturbation"] = _coerce_bool(params.get("run_perturbation", True), key="run_perturbation")
    params["generate_visualizations"] = _coerce_bool(
        params.get("generate_visualizations", True),
        key="generate_visualizations",
    )
    params["perturb_change"] = float(params.get("perturb_change", 0.5))
    params["direction_mode"] = str(params.get("direction_mode", "two_sided"))
    params["overall_perturbation_analysis"] = _coerce_bool(
        params.get("overall_perturbation_analysis", True),
        key="overall_perturbation_analysis",
    )
    params["perturbed_tracks"] = str(params.get("perturbed_tracks", "all"))
    perturb_stage = params.get("perturb_stage")
    if perturb_stage in (None, ""):
        params["perturb_stage"] = None
    else:
        params["perturb_stage"] = str(perturb_stage).strip()
    params["ignore_pathway_perturabtion"] = _coerce_bool(
        params.get("ignore_pathway_perturabtion", False),
        key="ignore_pathway_perturabtion",
    )
    params["ignore_drug_perturabtion"] = _coerce_bool(
        params.get("ignore_drug_perturabtion", False),
        key="ignore_drug_perturabtion",
    )
    params["centroid"] = _coerce_bool(params.get("centroid", False), key="centroid")
    if params.get("cpu_workers") not in (None, ""):
        params["cpu_workers"] = max(1, int(params.get("cpu_workers")))
    else:
        params["cpu_workers"] = None
    params["ignore_hcmarkers"] = _coerce_bool(params.get("ignore_hcmarkers", False), key="ignore_hcmarkers")
    params["ignore_dynamic_markers"] = _coerce_bool(
        params.get("ignore_dynamic_markers", False),
        key="ignore_dynamic_markers",
    )
    params["dynamic_marker_strategy"] = str(params.get("dynamic_marker_strategy", "optimized")).strip().lower()
    if params.get("dynamic_marker_cpu_workers") not in (None, ""):
        params["dynamic_marker_cpu_workers"] = max(1, int(params.get("dynamic_marker_cpu_workers")))
    else:
        params["dynamic_marker_cpu_workers"] = None
    params["dynamic_marker_force_serial"] = _coerce_bool(
        params.get("dynamic_marker_force_serial", False),
        key="dynamic_marker_force_serial",
    )
    params["dynamic_marker_verify_legacy"] = _coerce_bool(
        params.get("dynamic_marker_verify_legacy", False),
        key="dynamic_marker_verify_legacy",
    )
    params["customized_pathway"] = _resolve_optional_path(repo_root, params.get("customized_pathway"))
    params["cmap_dir"] = _resolve_optional_path(repo_root, params.get("cmap_dir"))

    if params["run_perturbation"]:
        customized_drug = params.get("customized_drug")
        if customized_drug in (None, ""):
            customized_drug_path = default_customized_drug_path.resolve()
        else:
            customized_drug_path = resolve_path(repo_root, str(customized_drug))
        if not params["ignore_drug_perturabtion"] and not customized_drug_path.exists():
            raise FileNotFoundError(f"Drug target database not found: {customized_drug_path}")
        params["customized_drug"] = str(customized_drug_path)
    else:
        params["customized_drug"] = None

    return params


def _pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _materialize_job_config(
    *,
    repo_root: Path,
    job_dir_path: Path,
    config_path: Optional[str],
    data_path: Optional[str],
    stage_key: Optional[str],
    total_stage: Optional[int],
    label_key: Optional[str],
    set_overrides: List[str],
) -> tuple[Path, Path]:
    source_cfg = resolve_default_config(repo_root, config_path)
    cfg = read_yaml(source_cfg)

    data_cfg = dict(cfg.get("data", {}))
    base_data_path = data_path if data_path is not None else str(data_cfg.get("path", "")).strip()
    if not base_data_path:
        raise ValueError("No data path provided and selected config does not define data.path")

    src_data = resolve_path(repo_root, base_data_path)
    isolated_input = stage_input_copy(src_data, job_dir_path / "input")

    data_cfg["path"] = str(isolated_input)
    if stage_key is not None:
        data_cfg["stage_key"] = stage_key
    if total_stage is not None:
        data_cfg["total_stage"] = int(total_stage)
    if label_key is not None:
        data_cfg["label_key"] = label_key
    cfg["data"] = data_cfg

    if set_overrides:
        cfg = apply_set_overrides(cfg, set_overrides)

    cfg_path = job_dir_path / "config.yaml"
    write_yaml(cfg_path, cfg)
    return cfg_path, source_cfg


def _materialize_job_config_payload(
    *,
    repo_root: Path,
    job_dir_path: Path,
    config_payload: Dict[str, Any],
    source_config_path: Optional[str],
) -> tuple[Path, Optional[Path]]:
    cfg = copy.deepcopy(config_payload or {})
    if not isinstance(cfg, dict):
        raise ValueError("Training config payload must be a mapping")

    data_cfg = dict(cfg.get("data", {}))
    base_data_path = str(data_cfg.get("path", "")).strip()
    if not base_data_path:
        raise ValueError("Training config payload is missing data.path")

    src_data = resolve_path(repo_root, base_data_path)
    isolated_input = stage_input_copy(src_data, job_dir_path / "input")
    data_cfg["path"] = str(isolated_input)
    cfg["data"] = data_cfg

    cfg_path = job_dir_path / "config.yaml"
    write_yaml(cfg_path, cfg)

    source_cfg = None
    if source_config_path:
        source_cfg = resolve_path(repo_root, source_config_path)
    return cfg_path, source_cfg


def _analysis_script_contents() -> str:
    return """#!/usr/bin/env python3
from __future__ import annotations

import inspect
import json
import sys
import traceback
from pathlib import Path

import yaml

REPO_ROOT = Path.cwd().resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from UNAGI.UNAGI_tool import UNAGI
from UNAGI.monitoring.idrem_summary import precompute_analysis_idrem_summaries
from UNAGI.monitoring.custom_perturbation import prewarm_metadata_cache
from UNAGI.monitoring.visualization_pipeline import generate_analysis_visualizations


def _write_visualization_status(target_dir: Path, *, status: str, manifest_path: Path | None = None, warning: str | None = None, error: str | None = None) -> None:
    payload = {
        "status": str(status),
        "manifest_path": str(manifest_path.resolve()) if manifest_path else None,
        "warning": str(warning) if warning else None,
        "error": str(error) if error else None,
    }
    status_path = target_dir / "visualization_status.json"
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: run_analysis.py <analysis_params.yaml>")

    params_path = Path(sys.argv[1]).resolve()
    params = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}

    data_path = Path(str(params["data_path"])).resolve()
    target_dir = Path(str(params["target_dir"])).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    customized_drug = params.get("customized_drug")
    if customized_drug in ("", None):
        customized_drug = None

    print(f"[ANALYSIS] data_path={data_path}", flush=True)
    print(f"[ANALYSIS] target_dir={target_dir}", flush=True)
    print(f"[ANALYSIS] iteration={int(params['iteration'])}", flush=True)

    try:
        unagi = UNAGI()
        analysis_kwargs = {
            "data_path": str(data_path),
            "iteration": int(params["iteration"]),
            "customized_pathway": params.get("customized_pathway"),
            "target_dir": str(target_dir),
            "customized_drug": customized_drug,
            "cmap_dir": params.get("cmap_dir"),
            "overall_perturbation_analysis": bool(params.get("overall_perturbation_analysis", True)),
            "perturbed_tracks": str(params.get("perturbed_tracks", "all")),
            "perturb_stage": params.get("perturb_stage"),
            "ignore_pathway_perturabtion": bool(params.get("ignore_pathway_perturabtion", False)),
            "ignore_drug_perturabtion": bool(params.get("ignore_drug_perturabtion", False)),
            "centroid": bool(params.get("centroid", False)),
            "ignore_hcmarkers": bool(params.get("ignore_hcmarkers", False)),
            "ignore_dynamic_markers": bool(params.get("ignore_dynamic_markers", False)),
        }
        signature = inspect.signature(unagi.analyse_UNAGI)
        supported = set(signature.parameters)

        if "random_background_sampling_times" in supported:
            analysis_kwargs["random_background_sampling_times"] = int(
                params.get("random_background_sampling_times", 20)
            )
        elif "progressionmarker_background_sampling_times" in supported:
            analysis_kwargs["progressionmarker_background_sampling_times"] = int(
                params.get("random_background_sampling_times", 20)
            )

        if "run_perturbation" in supported:
            analysis_kwargs["run_perturbation"] = bool(params.get("run_perturbation", True))
        elif "run_pertubration" in supported:
            analysis_kwargs["run_pertubration"] = bool(params.get("run_perturbation", True))

        if "perturb_change" in supported:
            analysis_kwargs["perturb_change"] = float(params.get("perturb_change", 0.5))
        elif "defulat_perturb_change" in supported:
            analysis_kwargs["defulat_perturb_change"] = float(params.get("perturb_change", 0.5))

        if "direction_mode" in supported:
            analysis_kwargs["direction_mode"] = str(params.get("direction_mode", "two_sided"))
        if "perturb_stage" in supported and params.get("perturb_stage") not in (None, ""):
            analysis_kwargs["perturb_stage"] = params.get("perturb_stage")

        if "perturbation_schedule" in supported and params.get("perturbation_schedule") is not None:
            analysis_kwargs["perturbation_schedule"] = str(params.get("perturbation_schedule"))
        if "cpu_workers" in supported and params.get("cpu_workers") is not None:
            analysis_kwargs["cpu_workers"] = int(params.get("cpu_workers"))
        if "dynamic_marker_strategy" in supported and params.get("dynamic_marker_strategy") is not None:
            analysis_kwargs["dynamic_marker_strategy"] = str(params.get("dynamic_marker_strategy"))
        if "dynamic_marker_cpu_workers" in supported and params.get("dynamic_marker_cpu_workers") is not None:
            analysis_kwargs["dynamic_marker_cpu_workers"] = int(params.get("dynamic_marker_cpu_workers"))
        if "dynamic_marker_force_serial" in supported:
            analysis_kwargs["dynamic_marker_force_serial"] = bool(params.get("dynamic_marker_force_serial", False))
        if "dynamic_marker_verify_legacy" in supported:
            analysis_kwargs["dynamic_marker_verify_legacy"] = bool(params.get("dynamic_marker_verify_legacy", False))

        unagi.analyse_UNAGI(**analysis_kwargs)
        print("[ANALYSIS] IDREM_SUMMARY START", flush=True)
        summary_result = precompute_analysis_idrem_summaries(target_dir)
        print(
            f"[ANALYSIS] IDREM_SUMMARY DONE tracks={summary_result.get('track_count', 0)} generated={len(summary_result.get('generated', []))}",
            flush=True,
        )
        if bool(params.get("generate_visualizations", True)):
            manifest_path = target_dir / "visualizations" / "manifest.json"
            print("[UNAGI-ANALYSIS] stage=generate_visualizations status=START", flush=True)
            _write_visualization_status(target_dir, status="running", manifest_path=manifest_path if manifest_path.exists() else None)
            try:
                manifest = generate_analysis_visualizations(target_dir)
                failure_count = int(((manifest.get("counts") or {}).get("failures") or 0))
                visualization_warning = None
                if failure_count:
                    visualization_warning = f"Visualization generation completed with {failure_count} failure(s)."
                _write_visualization_status(
                    target_dir,
                    status="succeeded",
                    manifest_path=manifest_path if manifest_path.exists() else None,
                    warning=visualization_warning,
                )
            except Exception as viz_exc:
                visualization_error = str(viz_exc)
                _write_visualization_status(target_dir, status="failed", error=visualization_error)
                print(f"[ANALYSIS] VISUALIZATION WARNING: {visualization_error}", flush=True)
                traceback.print_exc()
            print("[UNAGI-ANALYSIS] stage=generate_visualizations status=DONE", flush=True)
        else:
            _write_visualization_status(target_dir, status="pending")
        try:
            print("[ANALYSIS] PERTURBATION_METADATA_CACHE START", flush=True)
            metadata_cache = prewarm_metadata_cache(target_dir)
            track_count = len(((metadata_cache.get("form_context") or {}).get("track_options") or []))
            print(f"[ANALYSIS] PERTURBATION_METADATA_CACHE DONE tracks={track_count}", flush=True)
        except Exception as metadata_exc:
            print(f"[ANALYSIS] PERTURBATION_METADATA_CACHE WARNING: {metadata_exc}", flush=True)
            traceback.print_exc()
        print("[ANALYSIS] DONE", flush=True)
        return 0
    except Exception as exc:
        print(f"[ANALYSIS] ERROR: {exc}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
"""


def create_analysis_job(
    *,
    repo_root: Path,
    training_job_id: str,
    iteration: int,
    analysis_config_path: Optional[str],
    random_background_sampling_times: Optional[int],
    run_perturbation: Optional[bool],
    customized_pathway: Optional[str],
    customized_drug: Optional[str],
    cmap_dir: Optional[str],
    perturb_change: Optional[float],
    direction_mode: Optional[str],
    overall_perturbation_analysis: Optional[bool],
    perturbed_tracks: Optional[str],
    perturb_stage: Optional[str],
    ignore_pathway_perturabtion: Optional[bool],
    ignore_drug_perturabtion: Optional[bool],
    centroid: Optional[bool],
    cpu_workers: Optional[int],
    ignore_hcmarkers: Optional[bool],
    ignore_dynamic_markers: Optional[bool],
    set_overrides: List[str],
    max_retries: int,
    retry_delay_sec: int,
    job_name: Optional[str],
) -> Dict[str, Any]:
    source_jf = resolve_job_file(repo_root, training_job_id)
    source_job = read_json(source_jf)

    source_cfg = _resolve_existing_path(source_job.get("config_path"), source_jf.parent / "config.yaml").resolve()
    if not source_cfg.exists():
        raise FileNotFoundError(f"Source training config not found: {source_cfg}")
    source_dir = source_cfg.parent

    data_path = source_dir / "input" / str(int(iteration)) / "stagedata" / "dataset.h5ad"
    if not data_path.exists():
        raise FileNotFoundError(f"Iteration dataset not found: {data_path}")

    job_id = new_job_id("analyze")
    jd = source_dir / "analysis_jobs" / job_id
    jd.mkdir(parents=True, exist_ok=True)

    result_dir = source_dir / "analysis" / f"iter{int(iteration)}" / job_id / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    params = _analysis_defaults(iteration=int(iteration), data_path=data_path, target_dir=result_dir)
    analysis_template, analysis_template_path = _load_analysis_template(repo_root, analysis_config_path)
    if analysis_template:
        params.update(analysis_template)

    cli_overrides: Dict[str, Any] = {
        "random_background_sampling_times": random_background_sampling_times,
        "run_perturbation": run_perturbation,
        "customized_pathway": customized_pathway,
        "customized_drug": customized_drug,
        "cmap_dir": cmap_dir,
        "perturb_change": perturb_change,
        "direction_mode": direction_mode,
        "overall_perturbation_analysis": overall_perturbation_analysis,
        "perturbed_tracks": perturbed_tracks,
        "perturb_stage": perturb_stage,
        "ignore_pathway_perturabtion": ignore_pathway_perturabtion,
        "ignore_drug_perturabtion": ignore_drug_perturabtion,
        "centroid": centroid,
        "cpu_workers": cpu_workers,
        "ignore_hcmarkers": ignore_hcmarkers,
        "ignore_dynamic_markers": ignore_dynamic_markers,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            params[key] = value

    if set_overrides:
        params = apply_set_overrides(params, set_overrides)
    params = _normalize_analysis_params(
        params,
        repo_root=repo_root,
        iteration=int(iteration),
        data_path=data_path,
        target_dir=result_dir,
        default_customized_drug_path=(repo_root / "UNAGI" / "data" / "cmap_drug_target_genes.npy"),
    )

    params_path = jd / "analysis_params.yaml"
    write_yaml(params_path, params)

    script_path = jd / "run_analysis.py"
    script_path.write_text(_analysis_script_contents(), encoding="utf-8")

    log_path = jd / "analysis.log"
    jf = jd / "job.json"
    payload = {
        "job_id": job_id,
        "job_name": job_name or job_id,
        "job_kind": "analysis",
        "parent_job_id": training_job_id,
        "parent_job_file_path": str(source_jf),
        "status": "queued",
        "repo_root": str(repo_root),
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "started_at": None,
        "finished_at": None,
        "attempts": 0,
        "max_retries": max(0, int(max_retries)),
        "retry_delay_sec": max(1, int(retry_delay_sec)),
        "cancel_requested": False,
        "exit_code": None,
        "worker_pid": None,
        "child_pid": None,
        "source_config": str(source_cfg),
        "config_path": str(params_path),
        "log_path": str(log_path),
        "last_error": None,
        "command": None,
        "run_command": [sys.executable, "-u", str(script_path), str(params_path)],
        "analysis_iteration": int(params.get("iteration", iteration)),
        "analysis_output_dir": str(params.get("target_dir", str(result_dir))),
        "analysis_template_source": str(analysis_template_path) if analysis_template_path else None,
        "cpu_workers": params.get("cpu_workers"),
    }
    atomic_write_json(jf, payload)

    worker_pid, cmd = _launch_worker(repo_root, jf)
    payload["worker_pid"] = worker_pid
    payload["command"] = cmd
    payload["updated_at"] = utc_now_iso()
    atomic_write_json(jf, payload)
    return payload


def create_analysis_job_from_params(
    *,
    repo_root: Path,
    training_job_id: str,
    iteration: int,
    params_payload: Dict[str, Any],
    analysis_config_source: Optional[str] = None,
    max_retries: int = 0,
    retry_delay_sec: int = 60,
    job_name: Optional[str] = None,
) -> Dict[str, Any]:
    source_jf = resolve_job_file(repo_root, training_job_id)
    source_job = read_json(source_jf)

    source_cfg = _resolve_existing_path(source_job.get("config_path"), source_jf.parent / "config.yaml").resolve()
    if not source_cfg.exists():
        raise FileNotFoundError(f"Source training config not found: {source_cfg}")
    source_dir = source_cfg.parent

    data_path = source_dir / "input" / str(int(iteration)) / "stagedata" / "dataset.h5ad"
    if not data_path.exists():
        raise FileNotFoundError(f"Iteration dataset not found: {data_path}")

    job_id = new_job_id("analyze")
    jd = source_dir / "analysis_jobs" / job_id
    jd.mkdir(parents=True, exist_ok=True)

    result_dir = source_dir / "analysis" / f"iter{int(iteration)}" / job_id / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    params = _normalize_analysis_params(
        copy.deepcopy(params_payload or {}),
        repo_root=repo_root,
        iteration=int(iteration),
        data_path=data_path,
        target_dir=result_dir,
        default_customized_drug_path=(repo_root / "UNAGI" / "data" / "cmap_drug_target_genes.npy"),
    )

    params_path = jd / "analysis_params.yaml"
    write_yaml(params_path, params)

    script_path = jd / "run_analysis.py"
    script_path.write_text(_analysis_script_contents(), encoding="utf-8")

    log_path = jd / "analysis.log"
    jf = jd / "job.json"
    payload = {
        "job_id": job_id,
        "job_name": job_name or job_id,
        "job_kind": "analysis",
        "parent_job_id": training_job_id,
        "parent_job_file_path": str(source_jf),
        "status": "queued",
        "repo_root": str(repo_root),
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "started_at": None,
        "finished_at": None,
        "attempts": 0,
        "max_retries": max(0, int(max_retries)),
        "retry_delay_sec": max(1, int(retry_delay_sec)),
        "cancel_requested": False,
        "exit_code": None,
        "worker_pid": None,
        "child_pid": None,
        "source_config": str(source_cfg),
        "config_path": str(params_path),
        "log_path": str(log_path),
        "last_error": None,
        "command": None,
        "run_command": [sys.executable, "-u", str(script_path), str(params_path)],
        "analysis_iteration": int(params.get("iteration", iteration)),
        "analysis_output_dir": str(params.get("target_dir", str(result_dir))),
        "analysis_template_source": (
            str(resolve_path(repo_root, analysis_config_source)) if analysis_config_source else None
        ),
        "cpu_workers": params.get("cpu_workers"),
    }
    atomic_write_json(jf, payload)

    worker_pid, cmd = _launch_worker(repo_root, jf)
    payload["worker_pid"] = worker_pid
    payload["command"] = cmd
    payload["updated_at"] = utc_now_iso()
    atomic_write_json(jf, payload)
    return payload


def _perturbation_daemon_state_path(queue_key: str) -> Path:
    backend = _load_custom_perturbation_backend()
    filename = f"{backend._sanitize_queue_key(queue_key)}.json"
    return perturbation_worker_root() / filename


def _queue_daemon_payload(queue_key: str) -> dict[str, Any]:
    state_path = _perturbation_daemon_state_path(queue_key)
    if not state_path.exists():
        return {}
    try:
        return read_json(state_path)
    except Exception:
        return {}


def _write_queue_daemon_payload(queue_key: str, payload: dict[str, Any]) -> None:
    atomic_write_json(_perturbation_daemon_state_path(queue_key), payload)


def _queue_daemon_job_roots(*, repo_root: Path, queue_key: str, state: dict[str, Any] | None = None) -> list[Path]:
    payload = state or _queue_daemon_payload(queue_key)
    return _normalize_job_roots(repo_root, payload.get("watched_job_roots") or [])


def _register_queue_daemon_job_root(*, repo_root: Path, queue_key: str, job_root_path: Path | str) -> list[Path]:
    state = _queue_daemon_payload(queue_key)
    watched_roots = _normalize_job_roots(repo_root, list(state.get("watched_job_roots") or []) + [job_root_path])
    state["queue_key"] = str(queue_key)
    state["watched_job_roots"] = [str(path) for path in watched_roots]
    state["updated_at"] = utc_now_iso()
    _write_queue_daemon_payload(queue_key, state)
    return watched_roots


def _custom_perturbation_daemon_fingerprint() -> str:
    repo_root = infer_repo_root()
    fingerprint_paths = [
        Path(__file__).resolve(),
        (repo_root / "UNAGI" / "monitoring" / "custom_perturbation.py").resolve(),
        (repo_root / "UNAGI" / "perturbations" / "perturbation.py").resolve(),
        (repo_root / "UNAGI" / "perturbations" / "perturbation_centroid.py").resolve(),
        (repo_root / "UNAGI" / "perturbations" / "exact_parallel_executor.py").resolve(),
        (repo_root / "UNAGI" / "perturbations" / "analysis_perturbation.py").resolve(),
        (repo_root / "UNAGI" / "perturbations" / "normalize_analysis_perturbation.py").resolve(),
    ]
    payload: list[tuple[str, int | None, int | None]] = []
    for path in fingerprint_paths:
        if not path.exists():
            payload.append((str(path), None, None))
            continue
        stat = path.stat()
        payload.append((str(path), stat.st_mtime_ns, stat.st_size))
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _terminate_process(pid: int) -> None:
    if not _pid_alive(pid):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    for _ in range(10):
        if not _pid_alive(pid):
            return
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return
    for _ in range(10):
        if not _pid_alive(pid):
            return
        time.sleep(0.1)


def _requeue_active_custom_perturbation_jobs(
    *,
    repo_root: Path,
    queue_key: str,
    worker_pid: int,
    job_roots: Iterable[Path | str] | None = None,
) -> None:
    now = utc_now_iso()
    for job_file_path in _iter_custom_perturbation_jobs(repo_root, queue_key=queue_key, job_roots=job_roots):
        payload = read_json(job_file_path)
        if int(payload.get("worker_pid") or 0) != int(worker_pid):
            continue
        status = str(payload.get("status") or "")
        if status not in ACTIVE_STATUSES:
            continue
        if bool(payload.get("cancel_requested")) or status == "cancelling":
            payload.update(
                {
                    "status": "cancelled",
                    "finished_at": payload.get("finished_at") or now,
                    "updated_at": now,
                    "last_error": payload.get("last_error") or "Cancelled while restarting perturbation daemon.",
                    "worker_pid": None,
                    "child_pid": None,
                }
            )
        else:
            payload.update(
                {
                    "status": "queued",
                    "updated_at": now,
                    "last_error": "Requeued after perturbation daemon restart.",
                    "worker_pid": None,
                    "child_pid": None,
                    "started_at": None,
                    "attempt_started_at": None,
                    "exit_code": None,
                }
            )
        atomic_write_json(job_file_path, payload)


def _ensure_custom_perturbation_daemon(
    *,
    repo_root: Path,
    queue_key: str,
    watched_job_root: Path | str | None = None,
) -> tuple[int, List[str]]:
    state = _queue_daemon_payload(queue_key)
    watched_roots = _normalize_job_roots(
        repo_root,
        list(state.get("watched_job_roots") or []) + ([watched_job_root] if watched_job_root else []),
    )
    code_fingerprint = _custom_perturbation_daemon_fingerprint()
    existing_pid = int(state.get("pid") or 0)
    if _pid_alive(existing_pid):
        if str(state.get("code_fingerprint") or "") == code_fingerprint:
            command = list(state.get("command") or [])
            state.update(
                {
                    "queue_key": str(queue_key),
                    "watched_job_roots": [str(path) for path in watched_roots],
                    "updated_at": utc_now_iso(),
                }
            )
            if command:
                state["command"] = command
            _write_queue_daemon_payload(queue_key, state)
            return existing_pid, command
        _requeue_active_custom_perturbation_jobs(
            repo_root=repo_root,
            queue_key=queue_key,
            worker_pid=existing_pid,
            job_roots=watched_roots,
        )
        _terminate_process(existing_pid)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "perturb-daemon",
        "--repo-root",
        str(repo_root),
        "--queue-key",
        str(queue_key),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _write_queue_daemon_payload(
        queue_key,
        {
            "queue_key": str(queue_key),
            "pid": proc.pid,
            "command": cmd,
            "code_fingerprint": code_fingerprint,
            "watched_job_roots": [str(path) for path in watched_roots],
            "updated_at": utc_now_iso(),
        },
    )
    return proc.pid, cmd


def _custom_perturbation_signature_payload(payload: Dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": str(payload.get("mode") or ""),
        "scope_kind": str(payload.get("scope_kind") or ""),
        "track_key": payload.get("track_key"),
        "stage_key": payload.get("stage_key"),
        "perturb_change": float(payload.get("perturb_change") or 0.0),
        "direction_mode": str(payload.get("direction_mode") or ""),
        "requested_execution_mode": str(payload.get("requested_execution_mode") or ""),
        "execution_mode": str(payload.get("execution_mode") or ""),
        "resolved_execution_backend": str(payload.get("resolved_execution_backend") or ""),
        "target_spec": payload.get("target_spec") or {},
    }


def _custom_perturbation_signature(payload: Dict[str, Any]) -> str:
    return json.dumps(_custom_perturbation_signature_payload(payload), sort_keys=True)


def _find_duplicate_active_custom_perturbation_job(
    *,
    analysis_job_dir: Path,
    normalized_request: Dict[str, Any],
) -> Dict[str, Any] | None:
    perturbation_jobs_dir = analysis_job_dir / "perturbation_jobs"
    if not perturbation_jobs_dir.exists():
        return None
    target_signature = _custom_perturbation_signature(normalized_request)
    for job_file_path in sorted(
        perturbation_jobs_dir.glob("*/job.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    ):
        try:
            payload = read_json(job_file_path)
        except Exception:
            continue
        if str(payload.get("job_kind") or "") != "custom_perturbation":
            continue
        if str(payload.get("status") or "") not in ACTIVE_STATUSES:
            continue
        request_path = Path(str(payload.get("config_path") or job_file_path.parent / "request.json")).expanduser()
        if not request_path.exists():
            continue
        try:
            request_payload = read_json(request_path)
        except Exception:
            continue
        if _custom_perturbation_signature(request_payload) == target_signature:
            return payload
    return None


def create_custom_perturbation_job_from_payload(
    *,
    repo_root: Path,
    analysis_job_id: str,
    request_payload: Dict[str, Any],
    max_retries: int = 0,
    retry_delay_sec: int = 60,
    job_name: Optional[str] = None,
) -> Dict[str, Any]:
    analysis_jf = resolve_job_file(repo_root, analysis_job_id)
    return create_custom_perturbation_job_from_file(
        repo_root=repo_root,
        analysis_job_file_path=analysis_jf,
        analysis_job_id=analysis_job_id,
        request_payload=request_payload,
        max_retries=max_retries,
        retry_delay_sec=retry_delay_sec,
        job_name=job_name,
    )


def create_custom_perturbation_job_from_file(
    *,
    repo_root: Path,
    analysis_job_file_path: Path | str,
    analysis_job_id: Optional[str],
    request_payload: Dict[str, Any],
    max_retries: int = 0,
    retry_delay_sec: int = 60,
    job_name: Optional[str] = None,
) -> Dict[str, Any]:
    backend = _load_custom_perturbation_backend()
    analysis_jf = Path(analysis_job_file_path).expanduser().resolve()
    if not analysis_jf.exists():
        raise FileNotFoundError(f"Analysis job file not found: {analysis_jf}")
    analysis_job = read_json(analysis_jf)
    if str(analysis_job.get("job_kind")) != "analysis":
        raise ValueError("Custom perturbation jobs must be created from an analysis job")
    resolved_analysis_job_id = str(analysis_job_id or analysis_job.get("job_id") or analysis_jf.parent.name)

    analysis_output_dir = _resolve_analysis_output_dir_for_job(analysis_jf, analysis_job)
    if not analysis_output_dir.exists():
        raise FileNotFoundError(f"Analysis output directory not found: {analysis_output_dir}")

    validated = backend.validate_custom_perturbation_payload(
        analysis_dir=analysis_output_dir,
        analysis_job_id=resolved_analysis_job_id,
        analysis_iteration=int(analysis_job.get("analysis_iteration") or 0),
        raw_payload=request_payload,
    )
    normalized = dict(validated["normalized"])
    owning_job_root = _job_root_from_job_file(analysis_jf)
    duplicate_job = _find_duplicate_active_custom_perturbation_job(
        analysis_job_dir=analysis_jf.parent,
        normalized_request=normalized,
    )
    if duplicate_job is not None:
        worker_pid, cmd = _ensure_custom_perturbation_daemon(
            repo_root=repo_root,
            queue_key=str(normalized["queue_key"]),
            watched_job_root=owning_job_root,
        )
        duplicate_job_path = analysis_jf.parent / "perturbation_jobs" / str(duplicate_job["job_id"]) / "job.json"
        if duplicate_job_path.exists() and int(duplicate_job.get("worker_pid") or 0) != int(worker_pid):
            duplicate_job["worker_pid"] = worker_pid
            duplicate_job["command"] = cmd
            duplicate_job["updated_at"] = utc_now_iso()
            atomic_write_json(duplicate_job_path, duplicate_job)
        return duplicate_job

    job_id = new_job_id("perturb")
    analysis_job_dir = analysis_jf.parent
    job_dir_path = analysis_job_dir / "perturbation_jobs" / job_id
    job_dir_path.mkdir(parents=True, exist_ok=True)
    result_dir = analysis_output_dir / "custom_perturbations" / job_id
    result_dir.mkdir(parents=True, exist_ok=True)

    request_path = job_dir_path / "request.json"
    _write_json_payload = {
        **normalized,
        "analysis_dir": str(analysis_output_dir),
        "analysis_job_id": resolved_analysis_job_id,
    }
    request_path.write_text(json.dumps(_write_json_payload, indent=2), encoding="utf-8")

    log_path = job_dir_path / "perturbation.log"
    jf = job_dir_path / "job.json"
    resolved_job_name = job_name or normalized.get("user_label") or _default_custom_perturbation_job_name(normalized, job_id)
    payload = {
        "job_id": job_id,
        "job_name": resolved_job_name,
        "job_kind": "custom_perturbation",
        "parent_job_id": analysis_job_id,
        "status": "queued",
        "repo_root": str(repo_root),
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "started_at": None,
        "finished_at": None,
        "attempts": 0,
        "max_retries": max(0, int(max_retries)),
        "retry_delay_sec": max(1, int(retry_delay_sec)),
        "cancel_requested": False,
        "exit_code": None,
        "worker_pid": None,
        "child_pid": None,
        "source_config": str(analysis_job.get("config_path") or ""),
        "config_path": str(request_path),
        "log_path": str(log_path),
        "last_error": None,
        "command": None,
        "run_command": None,
        "analysis_iteration": int(analysis_job.get("analysis_iteration") or 0),
        "analysis_output_dir": str(result_dir),
        "queue_key": str(normalized["queue_key"]),
        "requested_execution_mode": str(normalized["requested_execution_mode"]),
        "execution_mode": str(normalized["execution_mode"]),
        "resolved_execution_backend": str(normalized.get("resolved_execution_backend") or normalized["execution_mode"]),
        "cpu_workers": int(normalized.get("cpu_workers") or 1),
        "scope_kind": str(normalized["scope_kind"]),
        "track_key": normalized.get("track_key"),
        "stage_key": normalized.get("stage_key"),
        "perturbation_mode": str(normalized["mode"]),
        "target_spec": normalized.get("target_spec"),
        "analysis_job_file_path": str(analysis_jf),
        "validation": {
            "estimated_scope_cell_count": int(validated["estimated_scope_cell_count"]),
            "analysis_n_obs": int(validated["analysis_n_obs"]),
            "warnings": list(validated["warnings"]),
        },
        "result_summary_path": None,
    }
    atomic_write_json(jf, payload)

    worker_pid, cmd = _ensure_custom_perturbation_daemon(
        repo_root=repo_root,
        queue_key=str(normalized["queue_key"]),
        watched_job_root=owning_job_root,
    )
    payload["worker_pid"] = worker_pid
    payload["command"] = cmd
    payload["updated_at"] = utc_now_iso()
    atomic_write_json(jf, payload)
    return payload


def _launch_worker(repo_root: Path, jf: Path) -> tuple[int, List[str]]:
    cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--job-file", str(jf)]
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc.pid, cmd


def create_job(
    *,
    repo_root: Path,
    config_path: Optional[str],
    data_path: Optional[str],
    stage_key: Optional[str],
    total_stage: Optional[int],
    label_key: Optional[str],
    set_overrides: List[str],
    max_retries: int,
    retry_delay_sec: int,
    job_name: Optional[str],
) -> Dict[str, Any]:
    job_id = new_job_id("train")
    jd = job_dir(repo_root, job_id)
    jd.mkdir(parents=True, exist_ok=True)

    cfg_path, source_cfg = _materialize_job_config(
        repo_root=repo_root,
        job_dir_path=jd,
        config_path=config_path,
        data_path=data_path,
        stage_key=stage_key,
        total_stage=total_stage,
        label_key=label_key,
        set_overrides=set_overrides,
    )

    log_path = jd / "training.log"
    jf = job_file(repo_root, job_id)
    payload = {
        "job_id": job_id,
        "job_name": job_name or job_id,
        "job_kind": "train",
        "status": "queued",
        "repo_root": str(repo_root),
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "started_at": None,
        "finished_at": None,
        "attempts": 0,
        "max_retries": max(0, int(max_retries)),
        "retry_delay_sec": max(1, int(retry_delay_sec)),
        "cancel_requested": False,
        "exit_code": None,
        "worker_pid": None,
        "child_pid": None,
        "source_config": str(source_cfg),
        "config_path": str(cfg_path),
        "log_path": str(log_path),
        "last_error": None,
        "command": None,
        "run_command": _child_command(str(cfg_path)),
    }
    atomic_write_json(jf, payload)

    worker_pid, cmd = _launch_worker(repo_root, jf)
    payload["worker_pid"] = worker_pid
    payload["command"] = cmd
    payload["updated_at"] = utc_now_iso()
    atomic_write_json(jf, payload)
    return payload


def create_job_from_config_payload(
    *,
    repo_root: Path,
    config_payload: Dict[str, Any],
    source_config_path: Optional[str] = None,
    max_retries: int = 0,
    retry_delay_sec: int = 60,
    job_name: Optional[str] = None,
) -> Dict[str, Any]:
    job_id = new_job_id("train")
    jd = job_dir(repo_root, job_id)
    jd.mkdir(parents=True, exist_ok=True)

    cfg_path, source_cfg = _materialize_job_config_payload(
        repo_root=repo_root,
        job_dir_path=jd,
        config_payload=config_payload,
        source_config_path=source_config_path,
    )

    log_path = jd / "training.log"
    jf = job_file(repo_root, job_id)
    payload = {
        "job_id": job_id,
        "job_name": job_name or job_id,
        "job_kind": "train",
        "status": "queued",
        "repo_root": str(repo_root),
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "started_at": None,
        "finished_at": None,
        "attempts": 0,
        "max_retries": max(0, int(max_retries)),
        "retry_delay_sec": max(1, int(retry_delay_sec)),
        "cancel_requested": False,
        "exit_code": None,
        "worker_pid": None,
        "child_pid": None,
        "source_config": str(source_cfg) if source_cfg else None,
        "config_path": str(cfg_path),
        "log_path": str(log_path),
        "last_error": None,
        "command": None,
        "run_command": _child_command(str(cfg_path)),
    }
    atomic_write_json(jf, payload)

    worker_pid, cmd = _launch_worker(repo_root, jf)
    payload["worker_pid"] = worker_pid
    payload["command"] = cmd
    payload["updated_at"] = utc_now_iso()
    atomic_write_json(jf, payload)
    return payload


def _child_command(config_path: str) -> List[str]:
    return [sys.executable, "-m", "UNAGI.cli", "run", "-c", config_path]


def _child_env() -> Dict[str, str]:
    env = dict(os.environ)
    numba_cache = env.get("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    Path(numba_cache).mkdir(parents=True, exist_ok=True)
    env["NUMBA_CACHE_DIR"] = numba_cache
    # In constrained environments (/dev/shm unavailable), these avoid semaphore-based multiprocessing init.
    env.setdefault("JOBLIB_MULTIPROCESSING", "0")
    env.setdefault("NUMBA_THREADING_LAYER", "workqueue")
    env.setdefault("NUMBA_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _update_job(jf: Path, mutator) -> Dict[str, Any]:
    payload = read_json(jf)
    mutator(payload)
    atomic_write_json(jf, payload)
    return payload


def _iter_custom_perturbation_jobs(
    repo_root: Path,
    *,
    queue_key: str,
    job_roots: Iterable[Path | str] | None = None,
) -> List[Path]:
    job_files: list[Path] = []
    seen: set[str] = set()
    for root in _normalize_job_roots(repo_root, job_roots):
        if not root.exists():
            continue
        for path in root.rglob("job.json"):
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            try:
                payload = read_json(path)
            except Exception:
                continue
            if str(payload.get("job_kind")) != "custom_perturbation":
                continue
            if str(payload.get("queue_key") or "") != str(queue_key):
                continue
            job_files.append(path)
    job_files.sort(key=lambda item: item.stat().st_mtime_ns)
    return job_files


def _next_custom_perturbation_job(
    repo_root: Path,
    *,
    queue_key: str,
    job_roots: Iterable[Path | str] | None = None,
) -> Path | None:
    for path in _iter_custom_perturbation_jobs(repo_root, queue_key=queue_key, job_roots=job_roots):
        payload = read_json(path)
        status = str(payload.get("status") or "")
        if payload.get("cancel_requested") and status in {"queued", "cancelling"}:
            atomic_write_json(
                path,
                {
                    **payload,
                    "status": "cancelled",
                    "finished_at": payload.get("finished_at") or utc_now_iso(),
                    "updated_at": utc_now_iso(),
                },
            )
            continue
        if status == "queued":
            return path
    return None


def _run_custom_perturbation_job(*, job_file_path: Path, queue_key: str, context_cache) -> int:
    backend = _load_custom_perturbation_backend()
    payload = read_json(job_file_path)
    attempts = int(payload.get("attempts") or 0)
    max_retries = int(payload.get("max_retries") or 0)
    retry_delay_sec = int(payload.get("retry_delay_sec") or 60)

    while attempts <= max_retries:
        payload = read_json(job_file_path)
        if bool(payload.get("cancel_requested")) and str(payload.get("status")) in {"queued", "cancelling"}:
            atomic_write_json(
                job_file_path,
                {
                    **payload,
                    "status": "cancelled",
                    "finished_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                },
            )
            return 130

        attempts += 1
        payload = _update_job(
            job_file_path,
            lambda p: p.update(
                {
                    "status": "running",
                    "attempts": attempts,
                    "worker_pid": os.getpid(),
                    "child_pid": os.getpid(),
                    "started_at": p.get("started_at") or utc_now_iso(),
                    "attempt_started_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                }
            ),
        )
        request_payload = read_json(Path(str(payload["config_path"])))
        result_dir = Path(str(payload["analysis_output_dir"])).expanduser().resolve()
        log_path = Path(str(payload["log_path"]))
        log_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(
                    f"\n[{utc_now_iso()}] attempt={attempts} queue={queue_key} custom_perturbation\n"
                )
                log_handle.flush()
                summary = backend.execute_custom_perturbation_job(
                    skill_root=skill_workspace_root(),
                    request_payload=request_payload,
                    result_dir=result_dir,
                    log_handle=log_handle,
                    context_cache=context_cache,
                )
        except Exception as exc:
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write("\n[UNAGI-CUSTOM-PERTURB] stage=error status=FAILED\n")
                log_handle.write(traceback.format_exc())
                log_handle.write("\n")
                log_handle.flush()
            payload = read_json(job_file_path)
            if attempts <= max_retries:
                atomic_write_json(
                    job_file_path,
                    {
                        **payload,
                        "status": "retrying",
                        "last_error": f"attempt {attempts} failed with error: {exc}",
                        "updated_at": utc_now_iso(),
                    },
                )
                time.sleep(max(1, retry_delay_sec))
                continue
            atomic_write_json(
                job_file_path,
                {
                    **payload,
                    "status": "failed",
                    "exit_code": 1,
                    "last_error": f"final attempt failed with error: {exc}",
                    "finished_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                },
            )
            return 1

        payload = read_json(job_file_path)
        atomic_write_json(
            job_file_path,
            {
                **payload,
                "status": "succeeded",
                "exit_code": 0,
                "result_summary_path": str(result_dir / "summary.json"),
                "finished_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
            },
        )
        return 0
    return 1


def run_custom_perturbation_daemon(*, repo_root: Path, queue_key: str) -> int:
    backend = _load_custom_perturbation_backend()
    context_cache = backend.PerturbationContextCache(skill_root=skill_workspace_root())
    state_path = _perturbation_daemon_state_path(queue_key)
    code_fingerprint = _custom_perturbation_daemon_fingerprint()
    existing_state = _queue_daemon_payload(queue_key)
    _write_queue_daemon_payload(
        queue_key,
        {
            "queue_key": str(queue_key),
            "pid": os.getpid(),
            "command": [sys.executable, str(Path(__file__).resolve()), "perturb-daemon", "--repo-root", str(repo_root), "--queue-key", str(queue_key)],
            "code_fingerprint": code_fingerprint,
            "watched_job_roots": [str(path) for path in _queue_daemon_job_roots(repo_root=repo_root, queue_key=queue_key, state=existing_state)],
            "updated_at": utc_now_iso(),
        },
    )

    stop_requested = False

    def _handle_signal(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        _write_queue_daemon_payload(
            queue_key,
            {
                "queue_key": str(queue_key),
                "pid": os.getpid(),
                "signal": int(signum),
                "code_fingerprint": code_fingerprint,
                "updated_at": utc_now_iso(),
            },
        )

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    while not stop_requested:
        state = _queue_daemon_payload(queue_key)
        watched_roots = _queue_daemon_job_roots(repo_root=repo_root, queue_key=queue_key, state=state)
        if state:
            state.update(
                {
                    "queue_key": str(queue_key),
                    "pid": os.getpid(),
                    "code_fingerprint": code_fingerprint,
                    "watched_job_roots": [str(path) for path in watched_roots],
                    "updated_at": utc_now_iso(),
                }
            )
            _write_queue_daemon_payload(queue_key, state)
        next_job = _next_custom_perturbation_job(repo_root, queue_key=queue_key, job_roots=watched_roots)
        if next_job is None:
            time.sleep(2)
            continue
        _run_custom_perturbation_job(job_file_path=next_job, queue_key=queue_key, context_cache=context_cache)

    try:
        state_path.unlink()
    except FileNotFoundError:
        pass
    return 0


def run_worker(job_file_path: Path) -> int:
    cancelled = False
    child: subprocess.Popen[str] | None = None

    def _handle_signal(signum, _frame):
        nonlocal cancelled, child
        cancelled = True
        _update_job(
            job_file_path,
            lambda p: p.update(
                {
                    "cancel_requested": True,
                    "last_signal": int(signum),
                    "updated_at": utc_now_iso(),
                }
            ),
        )
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    job = _update_job(
        job_file_path,
        lambda p: p.update(
            {
                "status": "running",
                "worker_pid": os.getpid(),
                "started_at": p.get("started_at") or utc_now_iso(),
                "updated_at": utc_now_iso(),
            }
        ),
    )

    max_retries = int(job.get("max_retries", 0))
    retry_delay_sec = int(job.get("retry_delay_sec", 60))
    attempts = int(job.get("attempts", 0))

    while attempts <= max_retries:
        attempts += 1
        job = _update_job(
            job_file_path,
            lambda p: p.update(
                {
                    "status": "running",
                    "attempts": attempts,
                    "attempt_started_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                }
            ),
        )

        run_command = job.get("run_command")
        if isinstance(run_command, list) and run_command:
            cmd = [str(part) for part in run_command]
        else:
            cmd = _child_command(str(job["config_path"]))
        log_path = Path(str(job["log_path"]))
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n[{utc_now_iso()}] attempt={attempts} cmd={' '.join(cmd)}\n")
            log_handle.flush()

            child = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(job["repo_root"]),
                env=_child_env(),
            )
            child_pid = child.pid
            _update_job(
                job_file_path,
                lambda p: p.update({"child_pid": child_pid, "command": cmd, "updated_at": utc_now_iso()}),
            )

            while child.poll() is None:
                latest = read_json(job_file_path)
                if cancelled or bool(latest.get("cancel_requested", False)):
                    cancelled = True
                    child.terminate()
                    break
                time.sleep(5)

            rc = child.wait()

        if cancelled:
            _update_job(
                job_file_path,
                lambda p: p.update(
                    {
                        "status": "cancelled",
                        "exit_code": rc,
                        "finished_at": utc_now_iso(),
                        "updated_at": utc_now_iso(),
                    }
                ),
            )
            return 130

        if rc == 0:
            _update_job(
                job_file_path,
                lambda p: p.update(
                    {
                        "status": "succeeded",
                        "exit_code": 0,
                        "finished_at": utc_now_iso(),
                        "updated_at": utc_now_iso(),
                    }
                ),
            )
            return 0

        if attempts <= max_retries:
            _update_job(
                job_file_path,
                lambda p: p.update(
                    {
                        "status": "retrying",
                        "last_error": f"attempt {attempts} failed with exit code {rc}",
                        "updated_at": utc_now_iso(),
                    }
                ),
            )
            time.sleep(max(1, retry_delay_sec))
            continue

        _update_job(
            job_file_path,
            lambda p: p.update(
                {
                    "status": "failed",
                    "exit_code": rc,
                    "last_error": f"final attempt failed with exit code {rc}",
                    "finished_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                }
            ),
        )
        return rc

    return 1


def get_job_status_payload(*, repo_root: Path, job_id: str) -> Dict[str, Any]:
    jf = resolve_job_file(repo_root, job_id)
    payload = read_json(jf)
    status = str(payload.get("status", "unknown"))
    worker_pid = int(payload.get("worker_pid") or 0)
    if status in ACTIVE_STATUSES and not _pid_alive(worker_pid):
        payload["status"] = "failed"
        payload["last_error"] = "worker process exited unexpectedly"
        payload["finished_at"] = payload.get("finished_at") or utc_now_iso()
        payload["updated_at"] = utc_now_iso()
        atomic_write_json(jf, payload)
        payload = read_json(jf)
    return payload


def list_jobs_payloads(
    *,
    repo_root: Path,
    status: Optional[str] = None,
    kind: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    files = sorted(job_root(repo_root).rglob("job.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    rows: List[Dict[str, Any]] = []
    for jf in files:
        try:
            payload = read_json(jf)
        except Exception:
            continue
        if status and str(payload.get("status")) != status:
            continue
        if kind and str(payload.get("job_kind", "train")) != kind:
            continue
        rows.append(
            {
                "job_id": payload.get("job_id"),
                "job_name": payload.get("job_name"),
                "job_kind": payload.get("job_kind", "train"),
                "parent_job_id": payload.get("parent_job_id"),
                "status": payload.get("status"),
                "attempts": payload.get("attempts"),
                "worker_pid": payload.get("worker_pid"),
                "created_at": payload.get("created_at"),
                "updated_at": payload.get("updated_at"),
                "log_path": payload.get("log_path"),
            }
        )
        if len(rows) >= max(1, int(limit)):
            break
    return rows


def cancel_job(*, repo_root: Path, job_id: str) -> Dict[str, Any]:
    jf = resolve_job_file(repo_root, job_id)
    return cancel_job_from_file(job_file_path=jf)


def cancel_job_from_file(*, job_file_path: Path | str) -> Dict[str, Any]:
    jf = Path(job_file_path).expanduser().resolve()
    if not jf.exists():
        raise FileNotFoundError(f"Unknown job file: {jf}")

    payload = read_json(jf)
    kind = str(payload.get("job_kind") or "train")
    job_id = str(payload.get("job_id") or jf.parent.name)
    payload["cancel_requested"] = True
    status = str(payload.get("status") or "")
    if kind == "custom_perturbation" and status in {"queued", "retrying"}:
        payload["status"] = "cancelled"
        payload["finished_at"] = payload.get("finished_at") or utc_now_iso()
    elif status in ACTIVE_STATUSES:
        payload["status"] = "cancelling"
    payload["updated_at"] = utc_now_iso()
    atomic_write_json(jf, payload)

    worker_pid = int(payload.get("worker_pid") or 0)
    signal_sent = False
    if kind != "custom_perturbation" and _pid_alive(worker_pid):
        try:
            os.kill(worker_pid, signal.SIGTERM)
            signal_sent = True
        except OSError:
            signal_sent = False

    return {
        "status": "cancel_requested",
        "job_id": job_id,
        "worker_pid": worker_pid,
        "signal_sent": signal_sent,
    }


def resume_job(
    *,
    repo_root: Path,
    job_id: str,
    resume_iteration: Optional[int] = None,
    max_retries: Optional[int] = None,
    retry_delay_sec: Optional[int] = None,
    set_overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    jf = resolve_job_file(repo_root, job_id)
    return resume_job_from_file(
        job_file_path=jf,
        repo_root=repo_root,
        resume_iteration=resume_iteration,
        max_retries=max_retries,
        retry_delay_sec=retry_delay_sec,
        set_overrides=set_overrides,
    )


def resume_job_from_file(
    *,
    job_file_path: Path | str,
    repo_root: Path | None = None,
    resume_iteration: Optional[int] = None,
    max_retries: Optional[int] = None,
    retry_delay_sec: Optional[int] = None,
    set_overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    resolved_repo_root = repo_root or infer_repo_root()
    jf = Path(job_file_path).expanduser().resolve()
    if not jf.exists():
        raise FileNotFoundError(f"Unknown job file: {jf}")

    payload = read_json(jf)
    job_id = str(payload.get("job_id") or jf.parent.name)
    if str(payload.get("status")) in ACTIVE_STATUSES:
        raise ValueError(f"Job {job_id} is currently active with status={payload.get('status')}")

    kind = str(payload.get("job_kind", "train"))
    cfg_path = Path(str(payload["config_path"]))
    if kind == "custom_perturbation":
        if resume_iteration is not None:
            raise ValueError("custom_perturbation jobs do not support resume_iteration")
        if set_overrides:
            raise ValueError("custom_perturbation jobs do not support config overrides on resume")
    else:
        cfg = read_yaml(cfg_path)
        if kind == "train":
            run_cfg = dict(cfg.get("run", {}))
            run_cfg["resume"] = True
            if resume_iteration is not None:
                run_cfg["resume_iteration"] = int(resume_iteration)
            cfg["run"] = run_cfg
        elif resume_iteration is not None:
            cfg["iteration"] = int(resume_iteration)
        if set_overrides:
            cfg = apply_set_overrides(cfg, set_overrides)
        if kind == "analysis":
            parent_job_id = str(payload.get("parent_job_id", "")).strip()
            if parent_job_id:
                parent_job_file_path = str(payload.get("parent_job_file_path") or "").strip()
                if parent_job_file_path:
                    parent_jf = Path(parent_job_file_path).expanduser().resolve()
                else:
                    candidate_parent = jf.parent.parents[2] / "job.json"
                    parent_jf = candidate_parent if candidate_parent.exists() else resolve_job_file(resolved_repo_root, parent_job_id)
                parent_payload = read_json(parent_jf)
                parent_cfg = Path(str(parent_payload.get("config_path", ""))).resolve()
                parent_dir = parent_cfg.parent
                iteration = int(cfg.get("iteration", payload.get("analysis_iteration", 0)))
                data_path = parent_dir / "input" / str(iteration) / "stagedata" / "dataset.h5ad"
                if not data_path.exists():
                    raise FileNotFoundError(f"Iteration dataset not found for resume: {data_path}")
                output_dir = parent_dir / "analysis" / f"iter{iteration}" / str(payload.get("job_id")) / "results"
                output_dir.mkdir(parents=True, exist_ok=True)
                cfg["data_path"] = str(data_path)
                cfg["target_dir"] = str(output_dir)
                payload["analysis_iteration"] = iteration
                payload["analysis_output_dir"] = str(output_dir)
        write_yaml(cfg_path, cfg)

    payload["status"] = "queued"
    payload["cancel_requested"] = False
    payload["attempts"] = 0
    payload["exit_code"] = None
    payload["last_error"] = None
    payload["started_at"] = None
    payload["finished_at"] = None
    payload["updated_at"] = utc_now_iso()
    if max_retries is not None:
        payload["max_retries"] = max(0, int(max_retries))
    if retry_delay_sec is not None:
        payload["retry_delay_sec"] = max(1, int(retry_delay_sec))
    atomic_write_json(jf, payload)

    if kind == "custom_perturbation":
        worker_pid, cmd = _ensure_custom_perturbation_daemon(
            repo_root=resolved_repo_root,
            queue_key=str(payload.get("queue_key") or "cpu:0"),
            watched_job_root=_job_root_from_job_file(jf),
        )
        payload["child_pid"] = None
    else:
        worker_pid, cmd = _launch_worker(resolved_repo_root, jf)
    payload["worker_pid"] = worker_pid
    payload["command"] = cmd
    payload["updated_at"] = utc_now_iso()
    atomic_write_json(jf, payload)

    return {
        "status": "resubmitted",
        "job_id": job_id,
        "worker_pid": worker_pid,
        "config_path": str(cfg_path),
        "log_path": payload.get("log_path"),
    }


def custom_perturbation_queue_diagnostics(*, repo_root: Path, job_file_path: Path | str) -> Dict[str, Any]:
    jf = Path(job_file_path).expanduser().resolve()
    if not jf.exists():
        raise FileNotFoundError(f"Unknown job file: {jf}")
    payload = read_json(jf)
    if str(payload.get("job_kind") or "") != "custom_perturbation":
        raise ValueError("Queue diagnostics are only available for custom perturbation jobs")

    queue_key = str(payload.get("queue_key") or "")
    state = _queue_daemon_payload(queue_key)
    daemon_pid = int(state.get("pid") or 0)
    daemon_alive = _pid_alive(daemon_pid)
    watched_roots = _queue_daemon_job_roots(repo_root=repo_root, queue_key=queue_key, state=state)
    watched_root_strings = [str(path) for path in watched_roots]
    job_root_path = _job_root_from_job_file(jf)
    root_watched = str(job_root_path) in watched_root_strings

    queue_position = None
    queued_job_ids: list[str] = []
    for position, candidate_path in enumerate(
        (
            path
            for path in _iter_custom_perturbation_jobs(repo_root, queue_key=queue_key, job_roots=watched_roots)
            if str(read_json(path).get("status") or "") == "queued"
        ),
        start=1,
    ):
        candidate_payload = read_json(candidate_path)
        candidate_job_id = str(candidate_payload.get("job_id") or candidate_path.parent.name)
        queued_job_ids.append(candidate_job_id)
        if candidate_path.resolve() == jf:
            queue_position = position

    status = str(payload.get("status") or "unknown")
    derived_status = status
    message = None
    if status == "queued":
        if not daemon_alive:
            derived_status = "stuck"
            message = "Queued, worker not running."
        elif not root_watched:
            derived_status = "stuck"
            message = "Queued, but no active worker is watching this run root."
        elif queue_position is not None:
            message = f"Queued on {queue_key}. Position {queue_position}."
        else:
            message = f"Queued on {queue_key}."
    elif status == "running":
        message = f"Running on {queue_key}."
    elif status == "retrying":
        message = f"Retrying on {queue_key}."

    return {
        "queue_key": queue_key,
        "daemon_pid": daemon_pid or None,
        "daemon_alive": daemon_alive,
        "job_root": str(job_root_path),
        "root_watched": root_watched,
        "watched_job_roots": watched_root_strings,
        "queue_position": queue_position,
        "queued_job_ids": queued_job_ids,
        "status": status,
        "derived_status": derived_status,
        "message": message,
    }


def tail_job_log(*, repo_root: Path, job_id: str, lines: int = 120) -> str:
    jf = resolve_job_file(repo_root, job_id)
    payload = read_json(jf)
    log_path = Path(str(payload["log_path"]))
    return _tail_lines(log_path, int(lines))


def _delete_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def _prune_empty_dirs(*, start: Optional[Path], stop: Path) -> list[str]:
    pruned: list[str] = []
    current = start.resolve() if start else None
    stop = stop.resolve()
    while current and current != stop:
        if not current.exists():
            current = current.parent
            continue
        try:
            next(current.iterdir())
            break
        except StopIteration:
            current.rmdir()
            pruned.append(str(current))
            current = current.parent
    return pruned


def _analysis_result_delete_target(payload: Dict[str, Any], job_dir_path: Path) -> Optional[Path]:
    output_value = str(payload.get("analysis_output_dir", "")).strip()
    if not output_value:
        return None
    output_path = Path(output_value).expanduser().resolve()
    if output_path.name == "results":
        return output_path.parent
    if output_path.parent.name == job_dir_path.name:
        return output_path.parent
    return output_path


def delete_job(*, repo_root: Path, job_id: str) -> Dict[str, Any]:
    jf = resolve_job_file(repo_root, job_id)
    return delete_job_from_file(job_file_path=jf)


def delete_job_from_file(*, job_file_path: Path | str) -> Dict[str, Any]:
    jf = Path(job_file_path).expanduser().resolve()
    if not jf.exists():
        raise FileNotFoundError(f"Unknown job file: {jf}")
    payload = read_json(jf)
    status = str(payload.get("status") or "").strip()
    if status not in TERMINAL_STATUSES:
        job_id = str(payload.get("job_id") or jf.parent.name)
        raise ValueError(f"Job {job_id} must be in a terminal state before deletion; current status={status or 'unknown'}")

    kind = str(payload.get("job_kind", "train"))
    job_id = str(payload.get("job_id") or jf.parent.name)
    deleted_paths: list[str] = []
    warnings: list[str] = []

    if kind == "train":
        target = jf.parent
        if not target.exists():
            warnings.append(f"Training job directory was already missing: {target}")
        else:
            _delete_path(target)
            deleted_paths.append(str(target))
        return {
            "status": "deleted",
            "job_id": job_id,
            "job_kind": kind,
            "deleted_paths": deleted_paths,
            "warnings": warnings,
        }

    if kind == "custom_perturbation":
        job_dir_path = jf.parent.resolve()
        analysis_job_dir = job_dir_path.parents[1]
        train_dir = job_dir_path.parents[3]
        result_value = str(payload.get("analysis_output_dir", "")).strip()
        result_dir = Path(result_value).expanduser().resolve() if result_value else None

        delete_targets: list[Path] = [job_dir_path]
        if result_dir is not None:
            delete_targets.insert(0, result_dir)

        for target in delete_targets:
            if not target.exists():
                warnings.append(f"Tracked path was already missing: {target}")
                continue
            _delete_path(target)
            deleted_paths.append(str(target))

        deleted_paths.extend(_prune_empty_dirs(start=job_dir_path.parent, stop=analysis_job_dir))
        if result_dir is not None:
            deleted_paths.extend(_prune_empty_dirs(start=result_dir.parent, stop=train_dir))

        return {
            "status": "deleted",
            "job_id": job_id,
            "job_kind": kind,
            "deleted_paths": deleted_paths,
            "warnings": warnings,
        }

    if kind != "analysis":
        raise ValueError(f"Unsupported job kind for deletion: {kind}")

    job_dir_path = jf.parent.resolve()
    train_dir = job_dir_path.parents[1]
    delete_targets: list[Path] = [job_dir_path]
    result_target = _analysis_result_delete_target(payload, job_dir_path)
    if result_target is not None and result_target not in delete_targets:
        delete_targets.insert(0, result_target)

    for target in delete_targets:
        if not target.exists():
            warnings.append(f"Tracked path was already missing: {target}")
            continue
        _delete_path(target)
        deleted_paths.append(str(target))

    deleted_paths.extend(_prune_empty_dirs(start=job_dir_path.parent, stop=train_dir))
    if result_target is not None:
        deleted_paths.extend(_prune_empty_dirs(start=result_target.parent, stop=train_dir))

    return {
        "status": "deleted",
        "job_id": job_id,
        "job_kind": kind,
        "deleted_paths": deleted_paths,
        "warnings": warnings,
    }


def delete_jobs(*, repo_root: Path, job_ids: Iterable[str]) -> Dict[str, Any]:
    deleted: list[Dict[str, Any]] = []
    blocked: list[Dict[str, Any]] = []
    failed: list[Dict[str, Any]] = []

    for raw_job_id in job_ids:
        job_id = str(raw_job_id).strip()
        if not job_id:
            continue
        try:
            deleted.append(delete_job(repo_root=repo_root, job_id=job_id))
        except FileNotFoundError as exc:
            failed.append({"job_id": job_id, "error": str(exc)})
        except ValueError as exc:
            blocked.append({"job_id": job_id, "error": str(exc)})
        except Exception as exc:
            failed.append({"job_id": job_id, "error": str(exc)})

    return {
        "status": "ok",
        "deleted": deleted,
        "blocked": blocked,
        "failed": failed,
    }


def delete_jobs_from_files(*, job_file_paths: Iterable[Path | str]) -> Dict[str, Any]:
    deleted: list[Dict[str, Any]] = []
    blocked: list[Dict[str, Any]] = []
    failed: list[Dict[str, Any]] = []

    for raw_path in job_file_paths:
        path = Path(raw_path).expanduser().resolve()
        job_id = path.parent.name
        try:
            deleted.append(delete_job_from_file(job_file_path=path))
        except FileNotFoundError as exc:
            failed.append({"job_id": job_id, "error": str(exc)})
        except ValueError as exc:
            blocked.append({"job_id": job_id, "error": str(exc)})
        except Exception as exc:
            failed.append({"job_id": job_id, "error": str(exc)})

    return {
        "status": "ok",
        "deleted": deleted,
        "blocked": blocked,
        "failed": failed,
    }


def cmd_submit(args: argparse.Namespace) -> int:
    repo_root = resolve_repo_root(args.repo_root)
    job = create_job(
        repo_root=repo_root,
        config_path=args.config,
        data_path=args.data_path,
        stage_key=args.stage_key,
        total_stage=args.total_stage,
        label_key=args.label_key,
        set_overrides=args.set or [],
        max_retries=args.max_retries,
        retry_delay_sec=args.retry_delay_sec,
        job_name=args.job_name,
    )
    print(
        json.dumps(
            {
                "status": "submitted",
                "job_id": job["job_id"],
                "job_name": job["job_name"],
                "worker_pid": job["worker_pid"],
                "config_path": job["config_path"],
                "log_path": job["log_path"],
            },
            indent=2,
        )
    )
    return 0


def cmd_submit_analysis(args: argparse.Namespace) -> int:
    repo_root = resolve_repo_root(args.repo_root)
    job = create_analysis_job(
        repo_root=repo_root,
        training_job_id=args.from_job,
        iteration=int(args.iteration),
        analysis_config_path=args.config,
        random_background_sampling_times=args.sampling_times,
        run_perturbation=args.run_perturbation,
        customized_pathway=args.customized_pathway,
        customized_drug=args.customized_drug,
        cmap_dir=args.cmap_dir,
        perturb_change=args.perturb_change,
        direction_mode=args.direction_mode,
        overall_perturbation_analysis=args.overall_perturbation_analysis,
        perturbed_tracks=args.perturbed_tracks,
        perturb_stage=args.perturb_stage,
        ignore_pathway_perturabtion=args.ignore_pathway_perturabtion,
        ignore_drug_perturabtion=args.ignore_drug_perturabtion,
        centroid=args.centroid,
        cpu_workers=args.cpu_workers,
        ignore_hcmarkers=args.ignore_hcmarkers,
        ignore_dynamic_markers=args.ignore_dynamic_markers,
        set_overrides=args.set or [],
        max_retries=int(args.max_retries),
        retry_delay_sec=int(args.retry_delay_sec),
        job_name=args.job_name,
    )
    print(
        json.dumps(
            {
                "status": "submitted",
                "job_id": job["job_id"],
                "job_name": job["job_name"],
                "job_kind": job.get("job_kind"),
                "parent_job_id": job.get("parent_job_id"),
                "analysis_iteration": job.get("analysis_iteration"),
                "worker_pid": job["worker_pid"],
                "config_path": job["config_path"],
                "log_path": job["log_path"],
                "analysis_output_dir": job.get("analysis_output_dir"),
                "analysis_template_source": job.get("analysis_template_source"),
            },
            indent=2,
        )
    )
    return 0


def cmd_submit_perturbation(args: argparse.Namespace) -> int:
    repo_root = resolve_repo_root(args.repo_root)
    request_payload = {
        "mode": args.mode,
        "scope_kind": args.scope_kind,
        "track_key": args.track_key,
        "stage_key": args.stage_key,
        "perturb_change": args.perturb_change,
        "direction_mode": args.direction_mode,
        "execution_mode": args.execution_mode,
        "cpu_workers": args.cpu_workers,
        "target_name": args.target_name,
        "gene_text": args.gene_text,
        "signed_gene_text": args.signed_gene_text,
        "user_label": args.user_label,
    }
    job = create_custom_perturbation_job_from_payload(
        repo_root=repo_root,
        analysis_job_id=args.analysis_job_id,
        request_payload=request_payload,
        max_retries=int(args.max_retries),
        retry_delay_sec=int(args.retry_delay_sec),
        job_name=args.job_name,
    )
    print(
        json.dumps(
            {
                "status": "submitted",
                "job_id": job["job_id"],
                "job_name": job["job_name"],
                "job_kind": job.get("job_kind"),
                "parent_job_id": job.get("parent_job_id"),
                "worker_pid": job.get("worker_pid"),
                "config_path": job.get("config_path"),
                "log_path": job.get("log_path"),
                "output_path": job.get("analysis_output_dir"),
                "execution_mode": job.get("execution_mode"),
                "resolved_execution_backend": job.get("resolved_execution_backend"),
                "cpu_workers": job.get("cpu_workers"),
            },
            indent=2,
        )
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    repo_root = resolve_repo_root(args.repo_root)
    payload = get_job_status_payload(repo_root=repo_root, job_id=args.job_id)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    repo_root = resolve_repo_root(args.repo_root)
    rows = list_jobs_payloads(
        repo_root=repo_root,
        status=args.status,
        kind=args.kind,
        limit=args.limit,
    )
    print(json.dumps(rows, indent=2))
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    repo_root = resolve_repo_root(args.repo_root)
    result = cancel_job(repo_root=repo_root, job_id=args.job_id)
    print(json.dumps(result, indent=2))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    repo_root = resolve_repo_root(args.repo_root)
    result = resume_job(
        repo_root=repo_root,
        job_id=args.job_id,
        resume_iteration=args.resume_iteration,
        max_retries=args.max_retries,
        retry_delay_sec=args.retry_delay_sec,
        set_overrides=args.set,
    )
    print(json.dumps(result, indent=2))
    return 0


def _tail_lines(path: Path, lines: int) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-max(1, lines) :])


def cmd_tail(args: argparse.Namespace) -> int:
    repo_root = resolve_repo_root(args.repo_root)
    print(tail_job_log(repo_root=repo_root, job_id=args.job_id, lines=int(args.lines)))
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    repo_root = resolve_repo_root(args.repo_root)
    result = delete_job(repo_root=repo_root, job_id=args.job_id)
    print(json.dumps(result, indent=2))
    return 0


def cmd_delete_many(args: argparse.Namespace) -> int:
    repo_root = resolve_repo_root(args.repo_root)
    result = delete_jobs(repo_root=repo_root, job_ids=args.job_id or [])
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UNAGI skill-first local job orchestration")
    parser.add_argument(
        "--skill-root",
        default=None,
        help="Override skill workspace root (parent containing skills/unagi-workflow-orchestrator)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    submit = sub.add_parser("submit", help="Submit a durable background UNAGI training job")
    submit.add_argument("--repo-root", default=None, help="UNAGI repository root")
    submit.add_argument("--config", default=None, help="Workflow YAML path (defaults to demo config)")
    submit.add_argument("--data-path", default=None, help="Input .h5ad file or directory")
    submit.add_argument("--stage-key", default=None, help="Override data.stage_key")
    submit.add_argument("--total-stage", type=int, default=None, help="Override data.total_stage")
    submit.add_argument("--label-key", default=None, help="Override data.label_key")
    submit.add_argument("--set", action="append", default=[], help="Override dotted.key=value (repeatable)")
    submit.add_argument("--max-retries", type=int, default=0, help="Maximum retry attempts")
    submit.add_argument("--retry-delay-sec", type=int, default=60, help="Retry delay in seconds")
    submit.add_argument("--job-name", default=None, help="Optional job display name")
    submit.set_defaults(func=cmd_submit)

    submit_analysis = sub.add_parser("submit-analysis", help="Submit a durable background UNAGI analysis job")
    submit_analysis.add_argument("--repo-root", default=None, help="UNAGI repository root")
    submit_analysis.add_argument(
        "--config",
        default=None,
        help="Analysis YAML template path (defaults to configs/unagi_analysis_template.yaml when present)",
    )
    submit_analysis.add_argument("--from-job", required=True, help="Source training job ID")
    submit_analysis.add_argument("--iteration", type=int, required=True, help="Iteration index to analyze")
    submit_analysis.add_argument("--sampling-times", type=int, default=None, help="Background sampling times")
    submit_analysis.add_argument("--customized-pathway", default=None, help="Path to customized pathway input")
    submit_analysis.add_argument("--customized-drug", default=None, help="Path to customized drug npy")
    submit_analysis.add_argument("--cmap-dir", default=None, help="Path to custom cmap directory")
    submit_analysis.add_argument("--perturb-change", type=float, default=None, help="Perturbation change level")
    submit_analysis.add_argument(
        "--direction-mode",
        choices=["two_sided", "one_sided"],
        default=None,
        help="Run perturbation analysis in two-sided (default) or one-sided mode",
    )
    submit_analysis.add_argument("--perturbed-tracks", default=None, help="Tracks to analyze")
    submit_analysis.add_argument("--perturb-stage", default=None, help="Limit pathway/drug scoring to one stage label or zero-based stage index")
    perturbation_mode = submit_analysis.add_mutually_exclusive_group()
    perturbation_mode.add_argument("--run-perturbation", action="store_true", dest="run_perturbation", help="Enable perturbation stage")
    perturbation_mode.add_argument("--no-perturbation", action="store_false", dest="run_perturbation", help="Disable perturbation stage")
    overall_mode = submit_analysis.add_mutually_exclusive_group()
    overall_mode.add_argument(
        "--overall-perturbation-analysis",
        action="store_true",
        dest="overall_perturbation_analysis",
        help="Compute perturbation analysis across all tracks",
    )
    overall_mode.add_argument(
        "--no-overall-perturbation-analysis",
        action="store_false",
        dest="overall_perturbation_analysis",
        help="Compute per-track perturbation analysis only",
    )
    submit_analysis.add_argument(
        "--ignore-pathway-perturabtion",
        action="store_true",
        default=None,
        help="Skip pathway perturbation task",
    )
    submit_analysis.add_argument(
        "--ignore-drug-perturabtion",
        action="store_true",
        default=None,
        help="Skip drug perturbation task",
    )
    submit_analysis.add_argument("--centroid", action="store_true", default=None, help="Use centroid perturbation mode")
    submit_analysis.add_argument("--cpu-workers", type=int, default=None, help="Override CPU worker count for exact CPU perturbation")
    submit_analysis.add_argument("--ignore-hcmarkers", action="store_true", default=None, help="Skip hierarchical marker discovery")
    submit_analysis.add_argument("--ignore-dynamic-markers", action="store_true", default=None, help="Skip dynamic marker discovery")
    submit_analysis.add_argument("--set", action="append", default=[], help="Override analysis dotted.key=value (repeatable)")
    submit_analysis.add_argument("--max-retries", type=int, default=0, help="Maximum retry attempts")
    submit_analysis.add_argument("--retry-delay-sec", type=int, default=60, help="Retry delay in seconds")
    submit_analysis.add_argument("--job-name", default=None, help="Optional job display name")
    submit_analysis.set_defaults(
        func=cmd_submit_analysis,
        run_perturbation=None,
        overall_perturbation_analysis=None,
    )

    submit_perturbation = sub.add_parser(
        "submit-perturbation",
        help="Submit a durable custom perturbation job under an analysis result",
    )
    submit_perturbation.add_argument("--repo-root", default=None, help="UNAGI repository root")
    submit_perturbation.add_argument("--analysis-job-id", required=True, help="Parent analysis job ID")
    submit_perturbation.add_argument("--mode", choices=["gene", "pathway", "drug"], required=True)
    submit_perturbation.add_argument("--scope-kind", choices=["overall", "track", "stage"], default="overall")
    submit_perturbation.add_argument("--track-key", default=None, help="Temporal track key when scope-kind=track")
    submit_perturbation.add_argument("--stage-key", default=None, help="Stage label when scope-kind=stage")
    submit_perturbation.add_argument("--perturb-change", type=float, default=0.5)
    submit_perturbation.add_argument("--direction-mode", choices=["two_sided", "one_sided"], default="two_sided")
    submit_perturbation.add_argument("--execution-mode", choices=["auto", "centroid", "cell_exact"], default="auto")
    submit_perturbation.add_argument("--cpu-workers", type=int, default=None, help="Override CPU worker count for exact CPU perturbation")
    submit_perturbation.add_argument("--target-name", default=None, help="Pathway or drug signature name")
    submit_perturbation.add_argument("--gene-text", default="", help="Comma or newline separated genes")
    submit_perturbation.add_argument("--signed-gene-text", default="", help="Comma or newline separated GENE:+ or GENE:- items")
    submit_perturbation.add_argument("--user-label", default=None, help="Optional display label")
    submit_perturbation.add_argument("--max-retries", type=int, default=0, help="Maximum retry attempts")
    submit_perturbation.add_argument("--retry-delay-sec", type=int, default=60, help="Retry delay in seconds")
    submit_perturbation.add_argument("--job-name", default=None, help="Optional job display name")
    submit_perturbation.set_defaults(func=cmd_submit_perturbation)

    status = sub.add_parser("status", help="Show full status payload for one job")
    status.add_argument("job_id", help="Job ID")
    status.add_argument("--repo-root", default=None, help="UNAGI repository root")
    status.set_defaults(func=cmd_status)

    list_cmd = sub.add_parser("list", help="List jobs with compact metadata")
    list_cmd.add_argument("--repo-root", default=None, help="UNAGI repository root")
    list_cmd.add_argument("--status", default=None, help="Filter by status")
    list_cmd.add_argument("--kind", choices=["train", "analysis", "custom_perturbation"], default=None, help="Filter by job kind")
    list_cmd.add_argument("--limit", type=int, default=50, help="Maximum rows")
    list_cmd.set_defaults(func=cmd_list)

    cancel = sub.add_parser("cancel", help="Request cancellation for a running job")
    cancel.add_argument("job_id", help="Job ID")
    cancel.add_argument("--repo-root", default=None, help="UNAGI repository root")
    cancel.set_defaults(func=cmd_cancel)

    resume = sub.add_parser("resume", help="Resume a failed or cancelled job")
    resume.add_argument("job_id", help="Job ID")
    resume.add_argument("--repo-root", default=None, help="UNAGI repository root")
    resume.add_argument("--resume-iteration", type=int, default=None, help="Optional run.resume_iteration")
    resume.add_argument("--max-retries", type=int, default=None, help="Override max retries")
    resume.add_argument("--retry-delay-sec", type=int, default=None, help="Override retry delay")
    resume.add_argument("--set", action="append", default=[], help="Override dotted.key=value (repeatable)")
    resume.set_defaults(func=cmd_resume)

    delete = sub.add_parser("delete", help="Permanently delete a terminal durable job")
    delete.add_argument("job_id", help="Job ID")
    delete.add_argument("--repo-root", default=None, help="UNAGI repository root")
    delete.set_defaults(func=cmd_delete)

    delete_many = sub.add_parser("delete-many", help="Permanently delete multiple terminal durable jobs")
    delete_many.add_argument("--repo-root", default=None, help="UNAGI repository root")
    delete_many.add_argument("--job-id", action="append", default=[], help="Job ID to delete (repeatable)")
    delete_many.set_defaults(func=cmd_delete_many)

    tail = sub.add_parser("tail", help="Print the last N lines from the job training log")
    tail.add_argument("job_id", help="Job ID")
    tail.add_argument("--repo-root", default=None, help="UNAGI repository root")
    tail.add_argument("--lines", type=int, default=120, help="Tail line count")
    tail.set_defaults(func=cmd_tail)

    worker = sub.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--job-file", required=True, help=argparse.SUPPRESS)

    perturb_daemon = sub.add_parser("perturb-daemon", help=argparse.SUPPRESS)
    perturb_daemon.add_argument("--repo-root", required=True, help=argparse.SUPPRESS)
    perturb_daemon.add_argument("--queue-key", required=True, help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "skill_root", None):
        os.environ[SKILL_WORKSPACE_ENV] = str(Path(args.skill_root).expanduser().resolve())

    if args.command == "worker":
        return run_worker(Path(args.job_file).resolve())
    if args.command == "perturb-daemon":
        return run_custom_perturbation_daemon(
            repo_root=resolve_repo_root(args.repo_root),
            queue_key=str(args.queue_key),
        )

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
