#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _run(cmd: List[str], cwd: Path) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def _parse_frontmatter(skill_md: Path) -> Dict[str, Any]:
    text = skill_md.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        raise ValueError("SKILL.md is missing YAML frontmatter")
    data = yaml.safe_load(match.group(1)) or {}
    if not isinstance(data, dict):
        raise ValueError("SKILL.md frontmatter must be a mapping")
    return data


def _resolve_quick_validate() -> Path:
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")).expanduser()
    quick_validate = codex_home / "skills" / ".system" / "skill-creator" / "scripts" / "quick_validate.py"
    if not quick_validate.exists():
        raise FileNotFoundError(f"quick_validate.py not found: {quick_validate}")
    return quick_validate


def evaluate_skill(skill_dir: Path, repo_root: Path) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    score = 100

    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        raise FileNotFoundError(f"Missing SKILL.md: {skill_md}")

    agents_yaml = skill_dir / "agents" / "openai.yaml"
    if not agents_yaml.exists():
        raise FileNotFoundError(f"Missing agents/openai.yaml: {agents_yaml}")

    frontmatter = _parse_frontmatter(skill_md)
    skill_name = str(frontmatter.get("name", "")).strip()
    description = str(frontmatter.get("description", "")).strip()
    body = skill_md.read_text(encoding="utf-8")

    ok_name = bool(re.match(r"^[a-z0-9-]+$", skill_name))
    checks.append({"check": "frontmatter_name_format", "ok": ok_name, "detail": skill_name})
    if not ok_name:
        score -= 20

    ok_description = "Use when" in description and len(description) >= 40
    checks.append({"check": "frontmatter_description_triggering", "ok": ok_description, "detail": description})
    if not ok_description:
        score -= 20

    no_todo = "TODO" not in body
    checks.append({"check": "no_template_todos", "ok": no_todo, "detail": "TODO markers absent"})
    if not no_todo:
        score -= 20

    agent_data = yaml.safe_load(agents_yaml.read_text(encoding="utf-8")) or {}
    interface = agent_data.get("interface", {}) if isinstance(agent_data, dict) else {}
    default_prompt = str(interface.get("default_prompt", "")).strip()
    ok_prompt = f"${skill_name}" in default_prompt
    checks.append({"check": "agent_default_prompt_mentions_skill", "ok": ok_prompt, "detail": default_prompt})
    if not ok_prompt:
        score -= 10

    quick_validate = _resolve_quick_validate()
    result_validate = _run([sys.executable, str(quick_validate), str(skill_dir)], cwd=repo_root)
    ok_validate = result_validate["returncode"] == 0
    checks.append(
        {
            "check": "quick_validate",
            "ok": ok_validate,
            "detail": result_validate["stdout"].strip() or result_validate["stderr"].strip(),
        }
    )
    if not ok_validate:
        score -= 15

    script_path = skill_dir / "scripts" / "unagi_job.py"
    env_guard_script = skill_dir / "scripts" / "check_python_env.py"
    help_checks = [
        [sys.executable, str(script_path), "--help"],
        [sys.executable, str(script_path), "submit", "--help"],
        [sys.executable, str(script_path), "submit-analysis", "--help"],
        [sys.executable, str(script_path), "status", "--help"],
        [sys.executable, str(script_path), "list", "--help"],
        [sys.executable, str(script_path), "cancel", "--help"],
        [sys.executable, str(script_path), "resume", "--help"],
        [sys.executable, str(script_path), "tail", "--help"],
    ]
    help_failures: List[str] = []
    for cmd in help_checks:
        out = _run(cmd, cwd=repo_root)
        if out["returncode"] != 0:
            help_failures.append(" ".join(cmd))
    ok_help = len(help_failures) == 0
    checks.append({"check": "job_script_help_accessible", "ok": ok_help, "detail": help_failures or "all help commands passed"})
    if not ok_help:
        score -= 15

    env_guard_exists = env_guard_script.exists()
    checks.append(
        {
            "check": "environment_guard_script_exists",
            "ok": env_guard_exists,
            "detail": str(env_guard_script),
        }
    )
    if not env_guard_exists:
        score -= 10

    env_guard_help_ok = False
    env_guard_help_detail = "script missing"
    if env_guard_exists:
        env_help = _run([sys.executable, str(env_guard_script), "--help"], cwd=repo_root)
        env_guard_help_ok = env_help["returncode"] == 0
        env_guard_help_detail = (
            "help command passed"
            if env_guard_help_ok
            else (env_help["stdout"].strip() or env_help["stderr"].strip())
        )
    checks.append(
        {
            "check": "environment_guard_help_accessible",
            "ok": env_guard_help_ok,
            "detail": env_guard_help_detail,
        }
    )
    if not env_guard_help_ok:
        score -= 10

    has_env_prompt = (
        "check_python_env.py" in body
        and "specify one virtual/conda environment" in body
    )
    checks.append(
        {
            "check": "skill_requires_environment_before_examples",
            "ok": has_env_prompt,
            "detail": "SKILL.md contains env check and explicit user prompt",
        }
    )
    if not has_env_prompt:
        score -= 10

    has_examples = "unagi_job.py submit" in body and "unagi_job.py status" in body
    checks.append({"check": "skill_contains_actionable_examples", "ok": has_examples, "detail": "submit and status examples in SKILL.md"})
    if not has_examples:
        score -= 10

    score = max(0, min(100, score))
    return {
        "skill_dir": str(skill_dir),
        "repo_root": str(repo_root),
        "score": score,
        "pass": score >= 90,
        "checks": checks,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate skill accessibility and command discoverability")
    parser.add_argument("--skill-dir", required=True, help="Path to skill directory")
    parser.add_argument("--repo-root", required=True, help="Path to UNAGI repo root")
    parser.add_argument("--output", required=True, help="Output JSON report path")
    args = parser.parse_args(argv)

    skill_dir = Path(args.skill_dir).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    report_path = Path(args.output).expanduser().resolve()

    report = evaluate_skill(skill_dir, repo_root)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
