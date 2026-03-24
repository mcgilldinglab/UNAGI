# UNAGI Skill Command Cheatsheet

## Environment Gate (Run First)

```bash
python skills/unagi-workflow-orchestrator/scripts/check_python_env.py
```

If the command fails, ask the user:
`Please specify one virtual/conda environment to use for UNAGI, then I will run the example test.`

## Short Tasks

```bash
python -m UNAGI.cli preflight -c configs/unagi_workflow_template.yaml
python -m UNAGI.cli run -c configs/unagi_workflow_template.yaml
```

## Long Tasks

```bash
python skills/unagi-workflow-orchestrator/scripts/unagi_job.py submit \
  --repo-root . \
  --config configs/unagi_workflow.example.run.yaml \
  --job-name baseline \
  --max-retries 1 \
  --retry-delay-sec 120

python skills/unagi-workflow-orchestrator/scripts/unagi_job.py submit-analysis \
  --repo-root . \
  --config configs/unagi_analysis_template.yaml \
  --from-job <training_job_id> \
  --iteration 2 \
  --sampling-times 20 \
  --job-name iter2-analysis

python skills/unagi-workflow-orchestrator/scripts/unagi_job.py list \
  --repo-root .

python skills/unagi-workflow-orchestrator/scripts/unagi_job.py status <job_id> \
  --repo-root .

python skills/unagi-workflow-orchestrator/scripts/unagi_job.py tail <job_id> \
  --repo-root . \
  --lines 150

python skills/unagi-workflow-orchestrator/scripts/unagi_job.py cancel <job_id> \
  --repo-root .

python skills/unagi-workflow-orchestrator/scripts/unagi_job.py resume <job_id> \
  --repo-root . \
  --resume-iteration 2

python skills/unagi-workflow-orchestrator/scripts/unagi_monitor_web.py serve \
  --repo-root . \
  --host 127.0.0.1 \
  --port 36871
```

## Override Training Parameters

```bash
python skills/unagi-workflow-orchestrator/scripts/unagi_job.py submit \
  --repo-root . \
  --config configs/unagi_workflow_template.yaml \
  --set training.epoch_iter=25 \
  --set training.lr=0.0001 \
  --set training.max_iter=8 \
  --set run.connect_edges_cutoff=0.04

python skills/unagi-workflow-orchestrator/scripts/unagi_job.py submit-analysis \
  --repo-root . \
  --from-job <training_job_id> \
  --iteration 2 \
  --config configs/unagi_analysis_template.yaml \
  --set perturb_change=0.35 \
  --set ignore_hcmarkers=true
```
