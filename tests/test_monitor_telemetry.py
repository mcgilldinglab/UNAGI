from __future__ import annotations

from types import SimpleNamespace

import UNAGI.monitoring.telemetry as telemetry


def test_collect_gpu_metrics_reports_gpus_and_process_mapping(monkeypatch):
    monkeypatch.setattr(telemetry.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def fake_run(cmd, check, capture_output, text):
        if "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu" in cmd:
            return SimpleNamespace(stdout="0, GPU-123, NVIDIA RTX, 42, 1024, 24576, 61\n")
        if "--query-compute-apps=pid,gpu_uuid,used_memory" in cmd:
            return SimpleNamespace(stdout="1234, GPU-123, 512\n")
        raise AssertionError(cmd)

    monkeypatch.setattr(telemetry.subprocess, "run", fake_run)

    gpus, pid_map, gpu_error = telemetry.collect_gpu_metrics()

    assert gpu_error is None
    assert gpus == [
        {
            "index": 0,
            "uuid": "GPU-123",
            "name": "NVIDIA RTX",
            "utilization_gpu_percent": 42.0,
            "memory_used_mb": 1024.0,
            "memory_total_mb": 24576.0,
            "temperature_c": 61.0,
        }
    ]
    assert pid_map == {1234: {"gpu_indices": [0], "memory_used_mb": 512.0}}


def test_collect_gpu_metrics_handles_missing_binary(monkeypatch):
    monkeypatch.setattr(telemetry.shutil, "which", lambda name: None)

    gpus, pid_map, gpu_error = telemetry.collect_gpu_metrics()

    assert gpus == []
    assert pid_map == {}
    assert gpu_error == "GPU telemetry unavailable: nvidia-smi not found"


def test_collect_process_metrics_includes_gpu_payload(monkeypatch):
    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def is_running(self):
            return True

        def oneshot(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cpu_percent(self, interval=None):
            return 7.5

        def memory_info(self):
            return SimpleNamespace(rss=4096)

        def status(self):
            return "running"

        def create_time(self):
            return 123.0

        def cmdline(self):
            return ["python", "worker.py"]

    monkeypatch.setattr(telemetry.psutil, "Process", FakeProcess)

    payload = telemetry.collect_process_metrics(321, {321: {"gpu_indices": [0], "memory_used_mb": 256.0}})

    assert payload == {
        "pid": 321,
        "cpu_percent": 7.5,
        "memory_rss_bytes": 4096,
        "status": "running",
        "create_time": 123.0,
        "cmdline": ["python", "worker.py"],
        "gpu": {"gpu_indices": [0], "memory_used_mb": 256.0},
    }
