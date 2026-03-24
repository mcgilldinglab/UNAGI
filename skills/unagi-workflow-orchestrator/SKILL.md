---
name: unagi-workflow-orchestrator
description: Orchestrate UNAGI workflows through local CLI jobs with durable submit/status/list/cancel/resume behavior. Use when the task is to run preflight checks, launch long UNAGI training (10 hours to multi-day), monitor progress, resume failed runs, or apply config overrides for .h5ad files or staged h5ad directories.
---

# UNAGI Workflow Orchestrator

## Overview

Run UNAGI as a local skill-first workflow, not a protocol server.
Use deterministic scripts for long jobs (training and analysis) and direct `unagi` CLI calls for short tasks.
Store staged data, logs, and outputs under the skill workspace root: the parent directory that contains `skills/unagi-workflow-orchestrator/`.
Resolve that workspace root dynamically from the actual skill location in use; never hard-code an absolute path.
The skill is intended to be installed locally from a cloned UNAGI checkout using your Codex/Claude skill installer, while still expecting the local UNAGI repo to exist as the workspace root.

## Prerequisites

- Require one active Python virtual environment before install checks or test examples. A conda environment is recommended, but a venv is acceptable.
- Run `python skills/unagi-workflow-orchestrator/scripts/check_python_env.py` first.
- If no virtual environment is detected, stop and ask the user to specify one virtual/conda environment, then continue only after activation.
- Prefer the bundled repo-local `idrem/` runtime automatically when it exists.
- If the user wants a different iDREM copy, set `run.idrem_dir` or `UNAGI_IDREM_DIR` to a directory that contains `idrem.jar`, `TFInput/`, and the rest of the iDREM runtime assets.
- Run commands from the UNAGI repository root.

## Workflow Routing

- Use direct CLI for short tasks:
  - `python -m UNAGI.cli preflight -c <config.yaml>`
  - `python -m UNAGI.cli run -c <config.yaml>`
- Use job orchestration for long tasks:
  - `python skills/unagi-workflow-orchestrator/scripts/unagi_job.py submit ...`
  - `python skills/unagi-workflow-orchestrator/scripts/unagi_job.py submit-analysis ...`
  - `python skills/unagi-workflow-orchestrator/scripts/unagi_job.py status <job_id>`
  - `python skills/unagi-workflow-orchestrator/scripts/unagi_job.py cancel <job_id>`
  - `python skills/unagi-workflow-orchestrator/scripts/unagi_job.py resume <job_id>`
- Use the local web console for durable jobs:
  - `python skills/unagi-workflow-orchestrator/scripts/unagi_monitor_web.py serve --repo-root .`

## Workspace Policy

- Keep durable job state under `outputs/skill_jobs/<job_id>` inside the skill workspace root.
- For ad hoc or example training runs, stage inputs and write results under `outputs/training_outputs/<run_name>/...` inside the skill workspace root.
- Run commands from the UNAGI repo root, but keep generated files under the skill workspace root.
- If the active skill is being run from an installed copy but you need a different workspace root, pass `--skill-root <path>` or set `UNAGI_SKILL_ROOT=<path>` before invoking `unagi_job.py`.

## Install And Example Test

```bash
# 1) Confirm one active environment exists before test execution
python skills/unagi-workflow-orchestrator/scripts/check_python_env.py

# 2) Run smoke-test preflight example from repo root
python -m UNAGI.cli preflight -c configs/unagi_workflow.example.run.yaml
```

If environment check fails, ask the user:
`Please specify one virtual/conda environment to use for UNAGI, then I will run the example test.`

## Quick Start

```bash
# Validate default demo run config
python -m UNAGI.cli preflight -c configs/unagi_workflow.example.run.yaml

# Submit a durable long job (defaults to demo config if --config omitted)
python skills/unagi-workflow-orchestrator/scripts/unagi_job.py submit \
  --repo-root . \
  --job-name demo-long-run

# Submit durable post-training analysis for one iteration from a finished training job
python skills/unagi-workflow-orchestrator/scripts/unagi_job.py submit-analysis \
  --repo-root . \
  --config configs/unagi_analysis_template.yaml \
  --from-job <training_job_id> \
  --iteration 2 \
  --job-name iter2-analysis

# Poll status
python skills/unagi-workflow-orchestrator/scripts/unagi_job.py list
python skills/unagi-workflow-orchestrator/scripts/unagi_job.py status <job_id>

# Launch the local web console on http://127.0.0.1:36871
python skills/unagi-workflow-orchestrator/scripts/unagi_monitor_web.py serve \
  --repo-root . \
  --host 127.0.0.1 \
  --port 36871

# Tail logs
python skills/unagi-workflow-orchestrator/scripts/unagi_job.py tail <job_id> --lines 120

# Cancel if needed
python skills/unagi-workflow-orchestrator/scripts/unagi_job.py cancel <job_id>

# Resume from latest checkpoint
python skills/unagi-workflow-orchestrator/scripts/unagi_job.py resume <job_id>
```

## Config Overrides

- Use `--set` for nested YAML overrides using dotted keys.
- Parse values as YAML scalars, so booleans and numbers work without manual casting.

```bash
python skills/unagi-workflow-orchestrator/scripts/unagi_job.py submit \
  --config configs/unagi_workflow_template.yaml \
  --set training.epoch_iter=30 \
  --set training.lr=0.0002 \
  --set run.connect_edges_cutoff=0.03

python skills/unagi-workflow-orchestrator/scripts/unagi_job.py submit-analysis \
  --from-job <training_job_id> \
  --iteration 2 \
  --config configs/unagi_analysis_template.yaml \
  --set perturb_change=0.4 \
  --set ignore_dynamic_markers=true
```

## Input Rules

- Accept `data.path` as a single `.h5ad` file or a directory containing `.h5ad` files.
- Stage inputs into per-job workspace under `outputs/skill_jobs/<job_id>/input` to avoid output collisions.
- Analysis outputs are grouped under the source training job, e.g. `outputs/skill_jobs/<training_job_id>/analysis/...`.
- For ad hoc or example runs outside `unagi_job.py`, stage data under `outputs/training_outputs/<run_name>/input` and keep training outputs in that same visible run directory.
- Default to `configs/unagi_workflow.example.run.yaml` when no config path is provided.
- Analysis defaults come from `configs/unagi_analysis_template.yaml` when present.

## Web Console Notes

- The web console is local/internal and read-write for durable jobs.
- It scans `outputs/skill_jobs/**/job.json` under the active skill workspace root.
- Use it to validate and submit training jobs, launch per-iteration analysis, cancel/resume jobs, and generate analysis visualizations.
- Training progress uses structured `[UNAGI-TRAIN]` markers when available and falls back to legacy `[UNAGI]` and `[TRAIN]` logs for older jobs.
- Analysis progress uses existing `[UNAGI-ANALYSIS]`, `[UNAGI-PERTURB]`, `[UNAGI-PROGRESSION]`, and `[ANALYSIS]` markers.
- Use `--data-root <path>` to allow additional server-side dataset/resource roots in the path picker.
- If you need to point the installed skill copy at a different workspace root, pass `--skill-root <path>` when starting the console.

## Accessibility Evaluation

Run the evaluator agent script after updates:

```bash
python skills/unagi-workflow-orchestrator/scripts/skill_accessibility_agent.py \
  --repo-root . \
  --skill-dir skills/unagi-workflow-orchestrator \
  --output skills/unagi-workflow-orchestrator/reports/accessibility_report.json
```

## References

- For detailed command patterns, read `references/command-cheatsheet.md`.
