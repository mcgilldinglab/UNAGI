from __future__ import annotations

import argparse
import json
import sys

from .workflow import PreflightError, run_workflow_from_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unagi", description="UNAGI workflow CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run UNAGI workflow from a YAML config")
    run_parser.add_argument("-c", "--config", required=True, help="Path to workflow YAML")

    preflight_parser = subparsers.add_parser("preflight", help="Validate workflow YAML and inputs")
    preflight_parser.add_argument("-c", "--config", required=True, help="Path to workflow YAML")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            result = run_workflow_from_config(args.config, validate_only=False)
        else:
            result = run_workflow_from_config(args.config, validate_only=True)
    except PreflightError as exc:
        print(f"[UNAGI preflight error] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - CLI safety
        print(f"[UNAGI error] {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
