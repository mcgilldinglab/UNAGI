from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

# Keep numba/joblib stable in restricted runtimes (e.g., no writable /dev/shm).
_numba_cache_dir = Path(os.environ.get("NUMBA_CACHE_DIR", "/tmp/numba_cache"))
_numba_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_numba_cache_dir))
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import scanpy as sc


warnings.filterwarnings("ignore", message="Series.__getitem__ treating keys as positions is deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message="In the future, the default backend for leiden will be igraph instead of leidenalg.*", category=FutureWarning)


class PreflightError(ValueError):
    """Raised when workflow inputs are invalid."""


def _stage_sort_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    try:
        return (0, float(text))
    except ValueError:
        return (1, text)


def _sorted_unique(values: Iterable[Any]) -> List[str]:
    unique_values = {str(v) for v in values}
    return sorted(unique_values, key=_stage_sort_key)


def _split_h5ad_to_stages(data_file: Path, output_dir: Path, stage_key: str, total_stage: int) -> Dict[str, int]:
    adata = sc.read_h5ad(data_file)
    if stage_key not in adata.obs.columns:
        raise PreflightError(
            f"Stage key {stage_key} not found in {data_file}. Available columns: {list(adata.obs.columns)}"
        )

    stage_labels = _sorted_unique(adata.obs[stage_key].astype(str).tolist())
    if len(stage_labels) != int(total_stage):
        raise PreflightError(
            f"Stage count mismatch: found {len(stage_labels)} unique {stage_key} values in {data_file}, "
            f"but total_stage={total_stage}."
        )

    mapping: Dict[str, int] = {}
    for idx, label in enumerate(stage_labels):
        subset = adata[adata.obs[stage_key].astype(str) == label].copy()
        subset.obs[stage_key] = str(idx)
        subset.obs["stage"] = str(idx)
        if "X_pca" not in subset.obsm.keys():
            sc.tl.pca(subset)
        subset.write_h5ad(output_dir / f"{idx}.h5ad", compression="gzip", compression_opts=9)
        mapping[label] = idx

    (output_dir / "stage_mapping.json").write_text(json.dumps(mapping, indent=2))
    return mapping


def normalize_stage_layout(data_path: str | Path, stage_key: str, total_stage: int) -> Path:
    """Normalize input data into canonical stage files: 0.h5ad..N-1.h5ad."""
    path = Path(data_path)
    if not path.exists():
        raise PreflightError(f"Data path does not exist: {path}")

    if path.is_file():
        output_dir = path.parent
        _split_h5ad_to_stages(path, output_dir, stage_key, total_stage)
        return output_dir

    canonical = [path / f"{i}.h5ad" for i in range(int(total_stage))]
    if all(p.exists() for p in canonical):
        return path

    h5ads = sorted(
        [p for p in path.glob("*.h5ad") if p.name != "dataset.h5ad"],
        key=lambda p: _stage_sort_key(p.stem),
    )
    if not h5ads:
        raise PreflightError(f"No .h5ad files found in {path}")

    if len(h5ads) == 1:
        _split_h5ad_to_stages(h5ads[0], path, stage_key, total_stage)
        return path

    if len(h5ads) != int(total_stage):
        raise PreflightError(
            f"Expected {total_stage} staged .h5ad files in {path}, found {len(h5ads)}. "
            "Provide either one combined .h5ad with a stage key, or exactly total_stage stage files."
        )

    mapping: Dict[str, int] = {}
    for idx, src in enumerate(h5ads):
        dst = path / f"{idx}.h5ad"
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        mapping[src.stem] = idx
    (path / "stage_mapping.json").write_text(json.dumps(mapping, indent=2))
    return path


def _is_valid_idrem_dir(path: Path) -> bool:
    return (
        path.exists()
        and path.is_dir()
        and (path / "idrem.jar").exists()
        and (path / "example_settings.txt").exists()
        and (path / "TFInput").is_dir()
    )


def _bundled_idrem_dirs() -> List[Path]:
    repo_root = Path(__file__).resolve().parent.parent
    return [
        repo_root / "third_party" / "idrem",
        repo_root / "idrem",
    ]


def _parse_scan_roots(value: Any) -> List[Path]:
    if value is None:
        return []
    if isinstance(value, str):
        return [Path(value)]
    if isinstance(value, list):
        roots: List[Path] = []
        for item in value:
            if item is None:
                continue
            roots.append(Path(str(item)))
        return roots
    raise PreflightError("run.idrem_scan_roots must be a string or list of strings")


def _add_candidate(candidates: List[Path], seen: set[Path], path: Path) -> None:
    expanded = path.expanduser()
    if expanded in seen:
        return
    seen.add(expanded)
    candidates.append(expanded)


def _candidate_idrem_dirs(configured: str | Path | None, data_dir: Path, extra_roots: List[Path]) -> List[Path]:
    candidates: List[Path] = []
    seen: set[Path] = set()

    for bundled in _bundled_idrem_dirs():
        _add_candidate(candidates, seen, bundled)

    if configured:
        configured_path = Path(configured).expanduser()
        _add_candidate(candidates, seen, configured_path)
        _add_candidate(candidates, seen, configured_path.parent)

    env_idrem = os.environ.get("UNAGI_IDREM_DIR")
    if env_idrem:
        env_path = Path(env_idrem).expanduser()
        _add_candidate(candidates, seen, env_path)
        _add_candidate(candidates, seen, env_path.parent)

    common_roots = [
        Path.cwd(),
        data_dir,
        data_dir.parent,
        Path.home(),
        Path.home() / "tools",
        Path("/opt"),
    ]
    for root in extra_roots:
        common_roots.append(root)

    for root in common_roots:
        _add_candidate(candidates, seen, root / "idrem")
        _add_candidate(candidates, seen, root / "UNAGI" / "idrem")
        _add_candidate(candidates, seen, root)

    return candidates


def _scan_for_idrem(candidates: List[Path], max_depth: int) -> Path | None:
    for candidate in candidates:
        if _is_valid_idrem_dir(candidate):
            return candidate

    for root in candidates:
        if not root.exists() or not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            current = Path(dirpath)
            try:
                rel_depth = len(current.relative_to(root).parts)
            except ValueError:
                rel_depth = max_depth + 1
            if rel_depth > max_depth:
                dirnames[:] = []
                continue

            if "idrem.jar" in filenames and "example_settings.txt" in filenames and "TFInput" in dirnames:
                return current

            if ".git" in dirnames:
                dirnames.remove(".git")

    return None


def _choose_default_install_dir(data_dir: Path) -> Path:
    env_install = os.environ.get("UNAGI_IDREM_INSTALL_DIR")
    if env_install:
        return Path(env_install).expanduser()
    for bundled in _bundled_idrem_dirs():
        if bundled.exists():
            return bundled
    return data_dir.parent / "idrem"


def resolve_idrem_dir(
    configured_idrem_dir: str | Path | None,
    data_dir: Path,
    *,
    auto_resolve: bool,
    auto_clone: bool,
    clone_url: str,
    install_dir: str | Path | None,
    scan_roots: List[Path],
    max_scan_depth: int,
) -> Path:
    configured_path = Path(configured_idrem_dir).expanduser() if configured_idrem_dir else None

    if configured_path is not None and _is_valid_idrem_dir(configured_path):
        return configured_path

    if configured_path is not None and configured_path.exists() and not auto_resolve:
        raise PreflightError(
            f"Configured run.idrem_dir is invalid: {configured_path}. "
            "Please provide an existing iDREM install containing idrem.jar, example_settings.txt, and TFInput/, "
            "or update UNAGI_IDREM_DIR to point at that install."
        )

    if auto_resolve:
        candidates = _candidate_idrem_dirs(configured_idrem_dir, data_dir, scan_roots)
        found = _scan_for_idrem(candidates, max_depth=max_scan_depth)
        if found is not None:
            return found

    if not auto_clone:
        raise PreflightError(
            "Could not resolve iDREM directory automatically. "
            "Please provide an existing iDREM install via run.idrem_dir or UNAGI_IDREM_DIR, "
            "or explicitly enable run.auto_clone_idrem if you want UNAGI to clone it for you."
        )

    target_dir = Path(install_dir).expanduser() if install_dir else _choose_default_install_dir(data_dir)
    if _is_valid_idrem_dir(target_dir):
        return target_dir

    if target_dir.exists() and any(target_dir.iterdir()):
        raise PreflightError(
            f"Cannot auto-clone iDREM into non-empty directory: {target_dir}. "
            "Please provide an empty run.idrem_install_dir or point run.idrem_dir / UNAGI_IDREM_DIR at an existing iDREM install."
        )

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", clone_url, str(target_dir)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise PreflightError(
            f"Auto-clone failed for iDREM from {clone_url} into {target_dir}: {exc.stderr.strip()}"
        ) from exc

    if not _is_valid_idrem_dir(target_dir):
        raise PreflightError(
            f"Auto-cloned iDREM directory is incomplete: {target_dir}. Expected idrem.jar, example_settings.txt, TFInput/."
        )

    return target_dir


def preflight_validate(
    data_path: str | Path,
    stage_key: str,
    total_stage: int,
    label_key: str = "name.simple",
    idrem_dir: str | Path | None = None,
    allow_existing_outputs: bool = False,
) -> Path:
    """Validate workflow inputs and normalize stage files."""
    data_dir = normalize_stage_layout(data_path, stage_key, total_stage)

    missing_stage_files = [str(i) for i in range(int(total_stage)) if not (data_dir / f"{i}.h5ad").exists()]
    if missing_stage_files:
        raise PreflightError(
            f"Missing canonical stage files in {data_dir}: {', '.join(missing_stage_files)}.h5ad"
        )

    sample = sc.read_h5ad(data_dir / "0.h5ad")
    if label_key not in sample.obs.columns:
        raise PreflightError(
            f"Label key {label_key} not found in {data_dir / '0.h5ad'}. "
            f"Available columns: {list(sample.obs.columns)}"
        )

    if idrem_dir is not None and not Path(idrem_dir).exists():
        raise PreflightError(
            f"iDREM directory does not exist: {idrem_dir}. "
            "Please provide a valid iDREM install path via run.idrem_dir or UNAGI_IDREM_DIR."
        )

    if not allow_existing_outputs:
        collisions = [p for p in [data_dir / "0", data_dir / "model_save"] if p.exists()]
        if collisions:
            joined = ", ".join(str(p) for p in collisions)
            raise PreflightError(
                f"Output folders already exist ({joined}). Use a fresh data directory or set "
                "preflight.allow_existing_outputs=true in workflow config."
            )

    return data_dir


def load_workflow_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise PreflightError(f"Workflow config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise PreflightError(f"Workflow config must be a mapping (dict): {path}")
    return cfg


def _save_workflow_config(config_path: str | Path, cfg: Dict[str, Any]) -> None:
    path = Path(config_path)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def run_workflow_from_payload(
    config_payload: Dict[str, Any],
    *,
    validate_only: bool = False,
    config_path: str | Path | None = None,
    persist_updates: bool = True,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(config_payload or {})
    if not isinstance(cfg, dict):
        raise PreflightError("Workflow config payload must be a mapping (dict)")

    data_cfg = cfg.get("data", {})
    training_cfg = cfg.get("training", {})
    run_cfg = cfg.get("run", {})
    preflight_cfg = cfg.get("preflight", {})

    data_path = data_cfg.get("path")
    stage_key = data_cfg.get("stage_key")
    total_stage = data_cfg.get("total_stage")
    label_key = data_cfg.get("label_key", "name.simple")

    if data_path is None:
        raise PreflightError("Config missing required field: data.path")
    if stage_key is None:
        raise PreflightError("Config missing required field: data.stage_key")
    if total_stage is None:
        raise PreflightError("Config missing required field: data.total_stage")

    data_dir = normalize_stage_layout(data_path, stage_key, int(total_stage))

    scan_roots = _parse_scan_roots(run_cfg.get("idrem_scan_roots"))
    idrem_dir_path = resolve_idrem_dir(
        configured_idrem_dir=run_cfg.get("idrem_dir"),
        data_dir=data_dir,
        auto_resolve=bool(run_cfg.get("auto_resolve_idrem", True)),
        auto_clone=bool(run_cfg.get("auto_clone_idrem", True)),
        clone_url=str(run_cfg.get("idrem_clone_url", "https://github.com/phoenixding/idrem")),
        install_dir=run_cfg.get("idrem_install_dir"),
        scan_roots=scan_roots,
        max_scan_depth=int(run_cfg.get("idrem_scan_depth", 4)),
    )
    idrem_dir = str(idrem_dir_path)

    if run_cfg.get("idrem_dir") != idrem_dir:
        run_cfg["idrem_dir"] = idrem_dir
        cfg["run"] = run_cfg
        if persist_updates and config_path and bool(run_cfg.get("persist_resolved_idrem_dir", True)):
            _save_workflow_config(config_path, cfg)

    preflight_validate(
        data_path=data_dir,
        stage_key=stage_key,
        total_stage=int(total_stage),
        label_key=label_key,
        idrem_dir=idrem_dir,
        allow_existing_outputs=bool(preflight_cfg.get("allow_existing_outputs", False)),
    )

    if validate_only:
        return {
            "status": "ok",
            "data_dir": str(data_dir),
            "stage_key": stage_key,
            "total_stage": int(total_stage),
            "idrem_dir": idrem_dir,
        }

    task = training_cfg.get("task")
    dist = training_cfg.get("dist")
    if task is None or dist is None:
        raise PreflightError("Config missing required fields: training.task and/or training.dist")

    requested_gpu = bool(training_cfg.get("GPU", False))
    selected_device = training_cfg.get("device")
    effective_gpu = requested_gpu
    if requested_gpu:
        try:
            import torch
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="CUDA initialization:.*", category=UserWarning)
                warnings.filterwarnings("ignore", message="Can't initialize NVML", category=UserWarning)
                cuda_ok = bool(torch.cuda.is_available()) and int(torch.cuda.device_count()) > 0
        except Exception:
            cuda_ok = False

        if not cuda_ok:
            print(
                "[UNAGI warning] GPU requested but CUDA is unavailable in this runtime. Falling back to CPU.",
                flush=True,
            )
            effective_gpu = False
            selected_device = "cpu"

    if not effective_gpu and (selected_device is None or str(selected_device).startswith("cuda")):
        selected_device = "cpu"

    from .UNAGI_tool import UNAGI

    unagi = UNAGI(random_seed=int(cfg.get("random_seed", 8848)))

    unagi.setup_data(
        data_path=str(data_dir),
        stage_key=stage_key,
        total_stage=int(total_stage),
        label_key=label_key,
        gcn_connectivities=bool(data_cfg.get("gcn_connectivities", False)),
        neighbors=int(data_cfg.get("neighbors", 25)),
        threads=int(data_cfg.get("threads", 20)),
        calculate_cell_graph=bool(data_cfg.get("calculate_cell_graph", True)),
        fast_mode=bool(data_cfg.get("fast_mode", True)),
    )

    unagi.setup_training(
        task=task,
        dist=dist,
        device=selected_device,
        epoch_iter=int(training_cfg.get("epoch_iter", 10)),
        epoch_initial=int(training_cfg.get("epoch_initial", 20)),
        lr=float(training_cfg.get("lr", 1e-4)),
        lr_dis=float(training_cfg.get("lr_dis", 5e-4)),
        beta=float(training_cfg.get("beta", 1)),
        hidden_dim=int(training_cfg.get("hidden_dim", 256)),
        latent_dim=int(training_cfg.get("latent_dim", 64)),
        graph_dim=int(training_cfg.get("graph_dim", 1024)),
        BATCHSIZE=int(training_cfg.get("BATCHSIZE", 512)),
        max_iter=int(training_cfg.get("max_iter", 10)),
        GPU=effective_gpu,
        adversarial=bool(training_cfg.get("adversarial", True)),
        GCN=bool(training_cfg.get("GCN", True)),
    )

    species = cfg.get("species")
    if species:
        unagi.register_species(species)

    cpo_cfg = cfg.get("CPO")
    if isinstance(cpo_cfg, dict):
        unagi.register_CPO_parameters(
            anchor_neighbors=int(cpo_cfg.get("anchor_neighbors", 15)),
            max_neighbors=int(cpo_cfg.get("max_neighbors", 35)),
            min_neighbors=int(cpo_cfg.get("min_neighbors", 10)),
            resolution_min=float(cpo_cfg.get("resolution_min", 0.8)),
            resolution_max=float(cpo_cfg.get("resolution_max", 1.5)),
        )

    idrem_cfg = cfg.get("iDREM")
    if isinstance(idrem_cfg, dict):
        unagi.register_iDREM_parameters(
            Normalize_data=idrem_cfg.get("Normalize_data", "Log_normalize_data"),
            Minimum_Absolute_Log_Ratio_Expression=float(
                idrem_cfg.get("Minimum_Absolute_Log_Ratio_Expression", 0.5)
            ),
            Convergence_Likelihood=float(idrem_cfg.get("Convergence_Likelihood", 0.001)),
            Minimum_Standard_Deviation=float(idrem_cfg.get("Minimum_Standard_Deviation", 0.5)),
        )

    unagi.run_UNAGI(
        idrem_dir=idrem_dir,
        CPO=bool(run_cfg.get("CPO", True)),
        resume=bool(run_cfg.get("resume", False)),
        resume_iteration=run_cfg.get("resume_iteration"),
        connect_edges_cutoff=float(run_cfg.get("connect_edges_cutoff", 0.05)),
    )

    return {
        "status": "ok",
        "task": task,
        "data_dir": str(data_dir),
        "max_iter": int(training_cfg.get("max_iter", 10)),
        "gpu_requested": requested_gpu,
        "gpu_effective": effective_gpu,
        "device": selected_device,
        "idrem_dir": idrem_dir,
    }


def run_workflow_from_config(config_path: str | Path, validate_only: bool = False) -> Dict[str, Any]:
    cfg = load_workflow_config(config_path)
    return run_workflow_from_payload(
        cfg,
        validate_only=validate_only,
        config_path=config_path,
        persist_updates=True,
    )
