'''
Helpers for dynamic-marker background sampling.

The legacy implementation shuffled each stage matrix sample-by-sample and then
recomputed per-cluster means by slicing AnnData views. The optimized path keeps
the same sampling semantics but caches stage-level dense matrices and cluster
row indices so the expensive AnnData work is not repeated in the inner loop.
'''
from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import math
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..dynamic_regulatory_networks.processIDREM import getClusterIdrem, getClusterPaths
from ..utils.progress_logger import ProgressLogger

_STAGE_CACHE: dict[str, dict[str, Any]] = {}
_PARALLEL_STAGE_CACHE: dict[str, Any] | None = None
_PARALLEL_SHUFFLE_PLANS: list[list[np.ndarray]] | None = None


def averageNode(nodes, state):
    '''
    calculate the average gene expression of sibling nodes
    args:
    nodes: number of sibling nodes
    state: the gene expression of each cluster in a certain stage

    return:
    out: the average gene expression of sibling nodes
    '''
    out = 0
    for each in nodes:
        out += state[each]
    return out / len(nodes)


def _hash_file_metadata(path: os.PathLike[str] | str | None) -> str:
    if path in (None, ''):
        return 'none'
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return f'missing:{resolved}'
    stat = resolved.stat()
    return f'{resolved}:{stat.st_size}:{int(stat.st_mtime_ns)}'


def _coerce_dense_matrix(matrix: Any) -> np.ndarray:
    if hasattr(matrix, 'toarray'):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def _write_stage_cache_manifest(cache_dir: Path | None, signature: str, stage_cache: dict[str, Any]) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / 'progressionmarker_stage_cache_manifest.json'
    payload = {
        'signature': signature,
        'stage_count': len(stage_cache['stages']),
        'total_stage': int(stage_cache['total_stage']),
        'track_count': len(stage_cache['track_names']),
        'times_independent': True,
        'stages': [
            {
                'stage': int(stage['stage']),
                'n_cells': int(stage['matrix'].shape[0]),
                'n_genes': int(stage['matrix'].shape[1]),
                'n_clusters': int(len(stage['cluster_indices'])),
            }
            for stage in stage_cache['stages']
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _stage_cache_signature(
    *,
    adata,
    total_stage: int,
    dataset_path: os.PathLike[str] | str | None,
    attribute_path: os.PathLike[str] | str | None,
) -> str:
    shape = tuple(int(x) for x in adata.shape)
    parts = [
        _hash_file_metadata(dataset_path),
        _hash_file_metadata(attribute_path),
        f'shape:{shape[0]}:{shape[1]}',
        f'total_stage:{int(total_stage)}',
    ]
    digest = hashlib.sha256('||'.join(parts).encode('utf-8')).hexdigest()
    return digest[:16]


def _build_track_names(paths: dict[str, list[list[int]]], total_stage: int) -> list[str]:
    track_names: list[str] = []
    path_list = [each for each in paths.values() if len(each) == total_stage]
    for each in path_list:
        track_name = ''
        for stage_idx, stage in enumerate(each):
            if stage_idx > 0:
                track_name += '-'
            for cluster_idx, cluster in enumerate(stage):
                if cluster_idx > 0:
                    track_name += 'n'
                track_name += str(cluster)
        track_names.append(track_name)
    return track_names


def _build_stage_cache(
    adata,
    total_stage: int,
    *,
    dataset_path: os.PathLike[str] | str | None = None,
    attribute_path: os.PathLike[str] | str | None = None,
    cache_dir: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    signature = _stage_cache_signature(
        adata=adata,
        total_stage=total_stage,
        dataset_path=dataset_path,
        attribute_path=attribute_path,
    )
    cached = _STAGE_CACHE.get(signature)
    if cached is not None:
        cached['cache_source'] = 'hit'
        _write_stage_cache_manifest(Path(cache_dir) if cache_dir is not None else None, signature, cached)
        return cached

    stage_cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
    stage_series = adata.obs['stage'].astype(str).to_numpy()
    stage_count = len(np.unique(stage_series))
    stages: list[dict[str, Any]] = []
    for stage_idx in range(stage_count):
        stage_mask = stage_series == str(stage_idx)
        stage_adata = adata[stage_mask]
        stage_dense = _coerce_dense_matrix(stage_adata.X)
        leiden_series = stage_adata.obs['leiden'].astype(str).to_numpy()
        cluster_count = len(np.unique(leiden_series))
        cluster_indices = [
            np.flatnonzero(leiden_series == str(cluster_idx)).astype(np.int32, copy=False)
            for cluster_idx in range(cluster_count)
        ]
        stages.append(
            {
                'stage': int(stage_idx),
                'matrix': stage_dense,
                'cluster_indices': cluster_indices,
            }
        )

    edges = {int(key): value for key, value in copy.deepcopy(adata.uns['edges']).items()}
    paths = getClusterPaths(edges, total_stage)
    stage_cache = {
        'signature': signature,
        'total_stage': int(total_stage),
        'stages': stages,
        'paths': paths,
        'track_names': _build_track_names(paths, total_stage),
        'cache_source': 'build',
    }
    _STAGE_CACHE[signature] = stage_cache
    _write_stage_cache_manifest(stage_cache_dir, signature, stage_cache)
    return stage_cache


def _build_shuffle_plans(stage_cache: dict[str, Any], times: int) -> list[list[np.ndarray]]:
    plans: list[list[np.ndarray]] = []
    for _sample_idx in range(times):
        sample_plan: list[np.ndarray] = []
        for stage in stage_cache['stages']:
            order = np.arange(stage['matrix'].shape[0], dtype=np.int32)
            np.random.shuffle(order)
            sample_plan.append(order)
        plans.append(sample_plan)
    return plans


def _compute_sample_result(stage_cache: dict[str, Any], shuffle_plan: list[np.ndarray]) -> dict[str, np.ndarray]:
    avg_expression_val: list[list[np.ndarray]] = []
    for stage, order in zip(stage_cache['stages'], shuffle_plan):
        shuffled = stage['matrix'][order]
        cluster_means = [
            np.mean(shuffled[cluster_indices], axis=0)
            for cluster_indices in stage['cluster_indices']
        ]
        avg_expression_val.append(cluster_means)

    idrem = np.asarray(getClusterIdrem(stage_cache['paths'], avg_expression_val, stage_cache['total_stage']))
    results: dict[str, np.ndarray] = {}
    for track_name, track_matrix in zip(stage_cache['track_names'], idrem):
        results[track_name] = track_matrix[:, -1] - track_matrix[:, 0]
    return results


def _compute_serial_results(
    stage_cache: dict[str, Any],
    shuffle_plans: list[list[np.ndarray]],
    *,
    logger: ProgressLogger,
    start_ts: float,
) -> dict[str, list[np.ndarray]]:
    results = {track_name: [None] * len(shuffle_plans) for track_name in stage_cache['track_names']}
    for sample_idx, shuffle_plan in enumerate(shuffle_plans):
        sample_result = _compute_sample_result(stage_cache, shuffle_plan)
        for track_name, delta in sample_result.items():
            results[track_name][sample_idx] = delta
        finished = sample_idx + 1
        if logger.should_report(finished, len(shuffle_plans)):
            logger.progress('progressionmarker_background', finished, len(shuffle_plans), start_ts)
    return {track_name: list(values) for track_name, values in results.items()}


def _dynamic_marker_worker_init() -> None:
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1)
    except Exception:
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ.setdefault(key, '1')


def _dynamic_marker_worker_compute(sample_range: tuple[int, int]) -> list[tuple[int, dict[str, np.ndarray]]]:
    if _PARALLEL_STAGE_CACHE is None or _PARALLEL_SHUFFLE_PLANS is None:
        raise RuntimeError('Dynamic-marker worker state is not configured.')
    start, end = sample_range
    out: list[tuple[int, dict[str, np.ndarray]]] = []
    for sample_idx in range(start, end):
        out.append((sample_idx, _compute_sample_result(_PARALLEL_STAGE_CACHE, _PARALLEL_SHUFFLE_PLANS[sample_idx])))
    return out


def _compute_parallel_results(
    stage_cache: dict[str, Any],
    shuffle_plans: list[list[np.ndarray]],
    *,
    cpu_workers: int,
    logger: ProgressLogger,
    start_ts: float,
) -> dict[str, list[np.ndarray]]:
    if len(shuffle_plans) == 0:
        return {track_name: [] for track_name in stage_cache['track_names']}

    global _PARALLEL_STAGE_CACHE, _PARALLEL_SHUFFLE_PLANS
    _PARALLEL_STAGE_CACHE = stage_cache
    _PARALLEL_SHUFFLE_PLANS = shuffle_plans
    results = {track_name: [None] * len(shuffle_plans) for track_name in stage_cache['track_names']}

    chunk_size = max(1, math.ceil(len(shuffle_plans) / max(1, cpu_workers * 2)))
    sample_ranges = [
        (start, min(start + chunk_size, len(shuffle_plans)))
        for start in range(0, len(shuffle_plans), chunk_size)
    ]

    try:
        mp_context = multiprocessing.get_context('fork')
    except ValueError:
        return _compute_serial_results(stage_cache, shuffle_plans, logger=logger, start_ts=start_ts)

    finished = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=cpu_workers,
        mp_context=mp_context,
        initializer=_dynamic_marker_worker_init,
    ) as executor:
        for shard in executor.map(_dynamic_marker_worker_compute, sample_ranges):
            for sample_idx, sample_result in shard:
                for track_name, delta in sample_result.items():
                    results[track_name][sample_idx] = delta
                finished += 1
                if logger.should_report(finished, len(shuffle_plans)):
                    logger.progress('progressionmarker_background', finished, len(shuffle_plans), start_ts)
    _PARALLEL_STAGE_CACHE = None
    _PARALLEL_SHUFFLE_PLANS = None
    return {track_name: list(values) for track_name, values in results.items()}


def _get_progressionmarker_background_legacy(times, adata, total_stage):
    results = {}
    edges = {int(key): value for key, value in copy.deepcopy(adata.uns['edges']).items()}
    paths = getClusterPaths(edges, total_stage)

    stage_data = []
    for i in range(len(adata.obs['stage'].unique())):
        adata.obs['stage'] = adata.obs['stage'].astype(str)
        temp = adata[adata.obs[adata.obs['stage'] == str(i)].index.tolist()]
        stage_data.append(temp)
    for _sample_idx in range(times):
        avg_expression_val = []
        for i in range(len(stage_data)):
            avg_expression_val.append([])
            tempX = stage_data[i].X.toarray()
            np.random.shuffle(tempX)
            stage_data[i].layers['shuffle_X'] = tempX
            for cluster in range(len(stage_data[i].obs['leiden'].unique())):
                avg_expression_val[i].append(
                    np.mean(
                        stage_data[i][stage_data[i].obs[stage_data[i].obs['leiden'] == str(cluster)].index.tolist()].layers[
                            'shuffle_X'
                        ],
                        axis=0,
                    )
                )
            del stage_data[i].layers['shuffle_X']

        idrem = np.array(getClusterIdrem(paths, avg_expression_val, total_stage))
        path = [each for each in paths.values() if len(each) == total_stage]
        for i, each in enumerate(path):
            track_name = ''
            for j, stage in enumerate(each):
                if j > 0:
                    track_name += '-'
                for k, cluster in enumerate(stage):
                    if k > 0:
                        track_name += 'n'
                    track_name += str(cluster)
            if track_name not in results.keys():
                results[track_name] = []
            results[track_name].append(idrem[i][:, -1] - idrem[i][:, 0])
    return results


def _normalize_dynamic_marker_strategy(strategy: str) -> str:
    value = str(strategy or 'optimized').strip().lower()
    if value not in {'legacy', 'optimized'}:
        raise ValueError(f'Unsupported dynamic-marker strategy: {strategy!r}')
    return value


def _normalize_dynamic_marker_workers(cpu_workers: int | None) -> int:
    if cpu_workers in (None, ''):
        return max(1, min(16, os.cpu_count() or 1))
    return max(1, int(cpu_workers))


def _compare_nested(
    left: Any,
    right: Any,
    *,
    atol: float,
    rtol: float,
    path: str = '',
) -> tuple[bool, str | None]:
    if isinstance(left, dict) and isinstance(right, dict):
        if list(left.keys()) != list(right.keys()):
            return False, f'{path}: key mismatch {list(left.keys())!r} != {list(right.keys())!r}'
        for key in left.keys():
            ok, detail = _compare_nested(left[key], right[key], atol=atol, rtol=rtol, path=f'{path}.{key}' if path else str(key))
            if not ok:
                return False, detail
        return True, None
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False, f'{path}: length mismatch {len(left)} != {len(right)}'
        for idx, (lval, rval) in enumerate(zip(left, right)):
            ok, detail = _compare_nested(lval, rval, atol=atol, rtol=rtol, path=f'{path}[{idx}]')
            if not ok:
                return False, detail
        return True, None
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not np.allclose(np.asarray(left), np.asarray(right), atol=atol, rtol=rtol, equal_nan=True):
            return False, f'{path}: array mismatch'
        return True, None
    if isinstance(left, (float, int, np.floating, np.integer)) and isinstance(right, (float, int, np.floating, np.integer)):
        if not np.isclose(float(left), float(right), atol=atol, rtol=rtol, equal_nan=True):
            return False, f'{path}: value mismatch {left!r} != {right!r}'
        return True, None
    if left != right:
        return False, f'{path}: value mismatch {left!r} != {right!r}'
    return True, None


def compare_progressionmarker_strategies(
    *,
    times: int,
    adata,
    total_stage: int,
    idrem_directory: os.PathLike[str] | str,
    size: int,
    cutoff: float = 0.05,
    dataset_path: os.PathLike[str] | str | None = None,
    attribute_path: os.PathLike[str] | str | None = None,
    cache_dir: os.PathLike[str] | str | None = None,
    cpu_workers: int | None = None,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> dict[str, Any]:
    from .dynamic_markers import runGetProgressionMarker_one_dist

    original_state = copy.deepcopy(np.random.get_state())
    try:
        np.random.set_state(copy.deepcopy(original_state))
        legacy_background = get_progressionmarker_background(
            times=times,
            adata=adata,
            total_stage=total_stage,
            strategy='legacy',
        )
        legacy_markers = runGetProgressionMarker_one_dist(
            str(idrem_directory),
            legacy_background,
            size,
            cutoff=cutoff,
            strategy='legacy',
        )

        np.random.set_state(copy.deepcopy(original_state))
        optimized_serial_background = get_progressionmarker_background(
            times=times,
            adata=adata,
            total_stage=total_stage,
            strategy='optimized',
            dataset_path=dataset_path,
            attribute_path=attribute_path,
            cache_dir=cache_dir,
            cpu_workers=1,
            force_serial=True,
        )
        optimized_serial_markers = runGetProgressionMarker_one_dist(
            str(idrem_directory),
            optimized_serial_background,
            size,
            cutoff=cutoff,
            strategy='optimized',
        )

        np.random.set_state(copy.deepcopy(original_state))
        optimized_parallel_background = get_progressionmarker_background(
            times=times,
            adata=adata,
            total_stage=total_stage,
            strategy='optimized',
            dataset_path=dataset_path,
            attribute_path=attribute_path,
            cache_dir=cache_dir,
            cpu_workers=cpu_workers,
            force_serial=False,
        )
        optimized_parallel_markers = runGetProgressionMarker_one_dist(
            str(idrem_directory),
            optimized_parallel_background,
            size,
            cutoff=cutoff,
            strategy='optimized',
        )
    finally:
        np.random.set_state(original_state)

    legacy_vs_serial_background = _compare_nested(
        legacy_background,
        optimized_serial_background,
        atol=atol,
        rtol=rtol,
    )
    legacy_vs_parallel_background = _compare_nested(
        legacy_background,
        optimized_parallel_background,
        atol=atol,
        rtol=rtol,
    )
    legacy_vs_serial_markers = _compare_nested(
        legacy_markers,
        optimized_serial_markers,
        atol=atol,
        rtol=rtol,
    )
    legacy_vs_parallel_markers = _compare_nested(
        legacy_markers,
        optimized_parallel_markers,
        atol=atol,
        rtol=rtol,
    )

    return {
        'ok': all(
            (
                legacy_vs_serial_background[0],
                legacy_vs_parallel_background[0],
                legacy_vs_serial_markers[0],
                legacy_vs_parallel_markers[0],
            )
        ),
        'comparisons': {
            'legacy_vs_optimized_serial_background': {
                'ok': legacy_vs_serial_background[0],
                'detail': legacy_vs_serial_background[1],
            },
            'legacy_vs_optimized_parallel_background': {
                'ok': legacy_vs_parallel_background[0],
                'detail': legacy_vs_parallel_background[1],
            },
            'legacy_vs_optimized_serial_markers': {
                'ok': legacy_vs_serial_markers[0],
                'detail': legacy_vs_serial_markers[1],
            },
            'legacy_vs_optimized_parallel_markers': {
                'ok': legacy_vs_parallel_markers[0],
                'detail': legacy_vs_parallel_markers[1],
            },
        },
        'legacy_background': legacy_background,
        'optimized_serial_background': optimized_serial_background,
        'optimized_parallel_background': optimized_parallel_background,
        'legacy_markers': legacy_markers,
        'optimized_serial_markers': optimized_serial_markers,
        'optimized_parallel_markers': optimized_parallel_markers,
    }


def get_progressionmarker_background(
    times,
    adata,
    total_stage,
    *,
    strategy: str = 'optimized',
    dataset_path: os.PathLike[str] | str | None = None,
    attribute_path: os.PathLike[str] | str | None = None,
    cache_dir: os.PathLike[str] | str | None = None,
    cpu_workers: int | None = None,
    force_serial: bool = False,
):
    '''
    sampling and simulate the random background for dynamic markers.
    args:
    times: the number of times to sample the random background
    adata: the single-cell data
    total_stage: the total number of time stages

    return:
    results: the simulated random background
    '''
    strategy = _normalize_dynamic_marker_strategy(strategy)
    logger = ProgressLogger(prefix='[UNAGI-PROGRESSION]')
    start_ts = time.time()
    logger.log(
        f"start background sampling: times={times}, total_stage={total_stage}, "
        f"strategy={strategy}, force_serial={bool(force_serial)}"
    )

    if strategy == 'legacy':
        results = _get_progressionmarker_background_legacy(times, adata, total_stage)
        logger.log('finished background sampling')
        return results

    stage_cache = _build_stage_cache(
        adata,
        total_stage,
        dataset_path=dataset_path,
        attribute_path=attribute_path,
        cache_dir=cache_dir,
    )
    logger.log(
        f"stage_cache source={stage_cache.get('cache_source', 'unknown')} "
        f"signature={stage_cache['signature']} tracks={len(stage_cache['track_names'])}"
    )
    shuffle_plans = _build_shuffle_plans(stage_cache, int(times))
    worker_count = _normalize_dynamic_marker_workers(cpu_workers)
    if force_serial or worker_count <= 1 or int(times) <= 1:
        results = _compute_serial_results(stage_cache, shuffle_plans, logger=logger, start_ts=start_ts)
    else:
        results = _compute_parallel_results(
            stage_cache,
            shuffle_plans,
            cpu_workers=worker_count,
            logger=logger,
            start_ts=start_ts,
        )
    logger.log('finished background sampling')
    return results
