#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
TRAIN_SCRIPT="$REPO_DIR/scripts/run_unagi_training_full.sh"

CONFIG_PATH="${1:-$REPO_DIR/configs/unagi_workflow.example.run.yaml}"
CRON_INTERVAL_MIN="${CRON_INTERVAL_MIN:-1}"
WATCH_INTERVAL_SEC="${WATCH_INTERVAL_SEC:-5}"
ALLOW_CONCURRENT_RUNS="${ALLOW_CONCURRENT_RUNS:-0}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[skill-train] missing config path: $CONFIG_PATH" >&2
  exit 2
fi

if [[ ! -x "$TRAIN_SCRIPT" ]]; then
  echo "[skill-train] missing executable training script: $TRAIN_SCRIPT" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[skill-train] nvidia-smi not found; cannot auto-select GPU" >&2
  exit 3
fi

BEST_GPU="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1" "$2}' \
  | sort -k2,2nr \
  | head -n 1 \
  | awk '{print $1}')"

if [[ -z "$BEST_GPU" ]]; then
  echo "[skill-train] failed to determine available GPU" >&2
  exit 4
fi

echo "[skill-train] selected GPU index=$BEST_GPU"
echo "[skill-train] launching training + monitor + watch"
echo "[skill-train] config=$CONFIG_PATH"
echo "[skill-train] tip: edit the config file before starting this script if needed"

exec env \
  CUDA_VISIBLE_DEVICES="$BEST_GPU" \
  CRON_INTERVAL_MIN="$CRON_INTERVAL_MIN" \
  WATCH_MONITOR=1 \
  WATCH_INTERVAL_SEC="$WATCH_INTERVAL_SEC" \
  ALLOW_CONCURRENT_RUNS="$ALLOW_CONCURRENT_RUNS" \
  COMPLETION_NOTIFY=1 \
  "$TRAIN_SCRIPT" "$CONFIG_PATH"
