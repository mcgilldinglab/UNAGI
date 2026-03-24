#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

try:
    import uvicorn
except Exception as exc:  # pragma: no cover - import error path
    raise SystemExit(f"uvicorn is required for the UNAGI monitor: {exc}") from exc

from unagi_job import resolve_repo_root, skill_workspace_root


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def _resolve_job_roots(values: Iterable[str]) -> list[Path]:
    roots: list[Path] = []
    for raw in values:
        path = Path(raw).expanduser().resolve()
        if path.name == "skill_jobs" and path.parent.name == "outputs":
            roots.append(path)
            continue
        roots.append(path / "outputs" / "skill_jobs")
    return _unique_paths(roots)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the UNAGI durable-job web console")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the local FastAPI monitor")
    serve.add_argument("--repo-root", default=None, help="UNAGI repository root")
    serve.add_argument(
        "--skill-root",
        default=None,
        help="Override skill workspace root (parent containing skills/unagi-workflow-orchestrator)",
    )
    serve.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve.add_argument("--port", type=int, default=36871, help="Bind port")
    serve.add_argument(
        "--scan-root",
        action="append",
        default=[],
        help="Additional workspace root or outputs/skill_jobs path to scan for jobs (repeatable)",
    )
    serve.add_argument(
        "--data-root",
        action="append",
        default=[],
        help="Additional allowed server-side data/resource root (repeatable)",
    )
    serve.set_defaults(func=cmd_serve)
    return parser


def cmd_serve(args: argparse.Namespace) -> int:
    if args.skill_root:
        os.environ["UNAGI_SKILL_ROOT"] = str(Path(args.skill_root).expanduser().resolve())
    repo_root = resolve_repo_root(args.repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from UNAGI.monitoring import create_monitor_app

    skill_root = skill_workspace_root()
    skill_dir = skill_root / "skills" / "unagi-workflow-orchestrator"
    job_roots = _resolve_job_roots(args.scan_root or [])
    data_roots = [Path(value).expanduser().resolve() for value in (args.data_root or [])]
    app = create_monitor_app(
        skill_root=skill_root,
        repo_root=repo_root,
        skill_dir=skill_dir,
        job_roots=job_roots,
        data_roots=data_roots or [repo_root, skill_root],
    )
    uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
