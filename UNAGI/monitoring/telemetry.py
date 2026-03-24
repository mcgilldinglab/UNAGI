from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import psutil


def _safe_cpu_percent() -> float | None:
    try:
        return float(psutil.cpu_percent(interval=None))
    except Exception:
        return None


def _safe_virtual_memory() -> dict | None:
    try:
        vm = psutil.virtual_memory()
    except Exception:
        return None
    return {
        "total_bytes": int(vm.total),
        "available_bytes": int(vm.available),
        "used_bytes": int(vm.used),
        "percent": float(vm.percent),
    }


def _safe_disk_usage(path: Path) -> dict | None:
    try:
        usage = psutil.disk_usage(str(path))
    except Exception:
        return None
    return {
        "path": str(path),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "percent": float(usage.percent),
    }


def collect_gpu_metrics() -> Tuple[List[dict], Dict[int, dict], str | None]:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return [], {}, "GPU telemetry unavailable: nvidia-smi not found"

    gpu_cmd = [
        binary,
        "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc_cmd = [
        binary,
        "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits",
    ]

    try:
        gpu_result = subprocess.run(gpu_cmd, check=True, capture_output=True, text=True)
    except Exception as exc:
        return [], {}, f"GPU telemetry unavailable: {exc}"

    gpus: List[dict] = []
    uuid_to_index: Dict[str, int] = {}
    for raw_line in gpu_result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        index = int(parts[0])
        gpu_uuid = parts[1]
        uuid_to_index[gpu_uuid] = index
        gpus.append(
            {
                "index": index,
                "uuid": gpu_uuid,
                "name": parts[2],
                "utilization_gpu_percent": float(parts[3]),
                "memory_used_mb": float(parts[4]),
                "memory_total_mb": float(parts[5]),
                "temperature_c": float(parts[6]),
            }
        )

    pid_map: Dict[int, dict] = {}
    try:
        proc_result = subprocess.run(proc_cmd, check=True, capture_output=True, text=True)
    except Exception:
        return gpus, pid_map, None

    for raw_line in proc_result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            gpu_uuid = parts[1]
            used_memory_mb = float(parts[2])
        except ValueError:
            continue
        entry = pid_map.setdefault(pid, {"gpu_indices": [], "memory_used_mb": 0.0})
        if gpu_uuid in uuid_to_index:
            gpu_index = uuid_to_index[gpu_uuid]
            if gpu_index not in entry["gpu_indices"]:
                entry["gpu_indices"].append(gpu_index)
        entry["memory_used_mb"] += used_memory_mb

    return gpus, pid_map, None


def collect_process_metrics(pid: int | None, gpu_pid_map: Dict[int, dict] | None = None) -> dict | None:
    if not pid:
        return None
    try:
        proc = psutil.Process(int(pid))
        if not proc.is_running():
            return None
        with proc.oneshot():
            cpu_percent = proc.cpu_percent(interval=None)
            memory = proc.memory_info()
            status = proc.status()
            create_time = proc.create_time()
            command = proc.cmdline()
    except Exception:
        return None

    payload = {
        "pid": int(pid),
        "cpu_percent": float(cpu_percent),
        "memory_rss_bytes": int(memory.rss),
        "status": status,
        "create_time": float(create_time),
        "cmdline": command,
    }
    if gpu_pid_map and int(pid) in gpu_pid_map:
        payload["gpu"] = gpu_pid_map[int(pid)]
    return payload


def collect_system_metrics(skill_root: Path) -> dict:
    gpus, gpu_pid_map, gpu_error = collect_gpu_metrics()
    return {
        "cpu_percent": _safe_cpu_percent(),
        "memory": _safe_virtual_memory(),
        "disk": _safe_disk_usage(skill_root),
        "gpus": gpus,
        "gpu_error": gpu_error,
        "gpu_pid_map": gpu_pid_map,
    }
