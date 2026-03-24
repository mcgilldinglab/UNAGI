from __future__ import annotations

import copy
import multiprocessing as mp
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import anndata as ad

from .perturbation import perturbation as ExactPerturbation


def default_cpu_workers(raw_value: Any = None) -> int:
    if raw_value not in (None, ""):
        return max(1, int(raw_value))
    return max(1, min(16, os.cpu_count() or 1))


def merge_directional_payloads(*payloads: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        for direction_key, track_payload in payload.items():
            if not isinstance(track_payload, Mapping):
                continue
            direction_bucket = merged.setdefault(str(direction_key), {})
            for track_key, item_payload in track_payload.items():
                if not isinstance(item_payload, Mapping):
                    continue
                track_bucket = direction_bucket.setdefault(str(track_key), {})
                for item_key, value in item_payload.items():
                    track_bucket[str(item_key)] = copy.deepcopy(value)
    return merged


def _read_analysis_adata(dataset_path: Path, attribute_path: Path | None):
    adata = ad.read_h5ad(dataset_path)
    if attribute_path is not None and attribute_path.exists():
        with attribute_path.open("rb") as handle:
            loaded_uns = pickle.load(handle)
        if isinstance(loaded_uns, dict):
            adata.uns.update(loaded_uns)
        else:
            adata.uns = loaded_uns
    return adata


def _set_cpu_thread_limits() -> None:
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass


def _set_backend_items(adata, *, backend_mode: str, items: Mapping[str, Sequence[str]]) -> None:
    if backend_mode == "pathway":
        adata.uns["data_pathway_overlap_genes"] = {
            str(key): list(value) for key, value in items.items()
        }
        adata.uns["pathway_gene_len_dict"] = {str(key): len(value) for key, value in items.items()}
    elif backend_mode == "drug":
        adata.uns["data_drug_overlap_genes"] = {
            str(key): list(value) for key, value in items.items()
        }
        adata.uns["drug-gene_len_dict"] = {str(key): len(value) for key, value in items.items()}
    else:
        raise ValueError(f"Unsupported backend_mode for exact parallel execution: {backend_mode}")


def _clear_runner_stage_keys(adata, *, backend_mode: str) -> None:
    keys = [f"{backend_mode}_perturbation_deltaD", f"{backend_mode}_perturbation_score"]
    for key in keys:
        adata.uns.pop(key, None)


_WORKER_STATE: dict[str, Any] = {}


def _worker_init(
    dataset_path: str,
    attribute_path: str | None,
    model_path: str,
    training_params_path: str,
    idrem_dir: str,
    backend_mode: str,
    perturb_change: float,
    direction_mode: str,
) -> None:
    _set_cpu_thread_limits()
    dataset = Path(dataset_path).expanduser().resolve()
    attribute = Path(attribute_path).expanduser().resolve() if attribute_path else None
    adata = _read_analysis_adata(dataset, attribute)
    runner = ExactPerturbation(
        adata,
        Path(model_path).expanduser().resolve(),
        Path(idrem_dir).expanduser().resolve(),
        config_path=Path(training_params_path).expanduser().resolve(),
        device_override="cpu",
    )
    _WORKER_STATE.clear()
    _WORKER_STATE.update(
        {
            "runner": runner,
            "backend_mode": backend_mode,
            "perturb_change": float(perturb_change),
            "direction_mode": str(direction_mode),
        }
    )


def _worker_run_subset(items: list[tuple[str, list[str]]]) -> dict[str, Any]:
    runner = _WORKER_STATE["runner"]
    backend_mode = _WORKER_STATE["backend_mode"]
    perturb_change = _WORKER_STATE["perturb_change"]
    direction_mode = _WORKER_STATE["direction_mode"]

    _clear_runner_stage_keys(runner.adata, backend_mode=backend_mode)
    _set_backend_items(runner.adata, backend_mode=backend_mode, items=dict(items))
    runner.run(
        backend_mode,
        float(perturb_change),
        inplace=True,
        CUDA=False,
        device="cpu",
        direction_mode=direction_mode,
    )
    return copy.deepcopy(runner.adata.uns.get(f"{backend_mode}_perturbation_deltaD", {}))


def _chunk_items(items: Mapping[str, Sequence[str]], worker_count: int) -> list[list[tuple[str, list[str]]]]:
    normalized = [(str(key), list(value)) for key, value in items.items()]
    if not normalized:
        return []
    worker_count = max(1, min(int(worker_count), len(normalized)))
    shards: list[list[tuple[str, list[str]]]] = [[] for _ in range(worker_count)]
    for index, item in enumerate(normalized):
        shards[index % worker_count].append(item)
    return [shard for shard in shards if shard]


def run_exact_parallel_delta(
    *,
    dataset_path: Path,
    attribute_path: Path | None,
    model_path: Path,
    training_params_path: Path,
    idrem_dir: Path,
    backend_mode: str,
    items: Mapping[str, Sequence[str]],
    perturb_change: float,
    direction_mode: str,
    cpu_workers: int | None = None,
) -> dict[str, Any]:
    worker_count = default_cpu_workers(cpu_workers)
    shards = _chunk_items(items, worker_count)
    if not shards:
        return {}
    if len(shards) == 1:
        _worker_init(
            str(dataset_path),
            str(attribute_path) if attribute_path else None,
            str(model_path),
            str(training_params_path),
            str(idrem_dir),
            backend_mode,
            float(perturb_change),
            str(direction_mode),
        )
        return _worker_run_subset(shards[0])

    available_methods = set(mp.get_all_start_methods())
    context_name = "fork" if "fork" in available_methods else "spawn"
    mp_context = mp.get_context(context_name)
    with ProcessPoolExecutor(
        max_workers=len(shards),
        mp_context=mp_context,
        initializer=_worker_init,
        initargs=(
            str(dataset_path),
            str(attribute_path) if attribute_path else None,
            str(model_path),
            str(training_params_path),
            str(idrem_dir),
            backend_mode,
            float(perturb_change),
            str(direction_mode),
        ),
    ) as executor:
        payloads = [future.result() for future in [executor.submit(_worker_run_subset, shard) for shard in shards]]
    return merge_directional_payloads(*payloads)
