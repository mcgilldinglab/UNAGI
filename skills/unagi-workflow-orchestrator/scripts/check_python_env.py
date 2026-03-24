#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple


EnvInfo = Tuple[str, str]


def detect_env() -> Optional[EnvInfo]:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "").strip()
    venv_path = os.environ.get("VIRTUAL_ENV", "").strip()

    if conda_env:
        return ("conda", conda_env)
    if venv_path:
        return ("venv", venv_path)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Require an active Python virtual environment before running UNAGI examples"
    )
    parser.add_argument(
        "--require-conda",
        action="store_true",
        help="Require Conda environment specifically",
    )
    args = parser.parse_args(argv)

    info = detect_env()
    if info is None:
        print("No active Python virtual environment detected.", file=sys.stderr)
        print(
            "Please provide one virtual/conda environment to use before running UNAGI.",
            file=sys.stderr,
        )
        print(
            "UNAGI does not create or manage the Python environment for you.",
            file=sys.stderr,
        )
        print("Example: conda activate <env-name>", file=sys.stderr)
        print("Example: source <venv-path>/bin/activate", file=sys.stderr)
        return 2

    kind, name = info
    if args.require_conda and kind != "conda":
        print(
            "A virtual environment is active but not a Conda environment.",
            file=sys.stderr,
        )
        print(
            "Please provide one Conda environment to use before running UNAGI.",
            file=sys.stderr,
        )
        print(
            "UNAGI does not switch environments automatically.",
            file=sys.stderr,
        )
        return 3

    if kind == "conda":
        print(f"Conda environment detected: {name}")
    else:
        print(f"Virtual environment detected: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
