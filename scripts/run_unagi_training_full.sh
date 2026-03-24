#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-unagi}"
MONITOR_SCRIPT="$REPO_DIR/scripts/monitor_unagi_passive.sh"
WATCH_MONITOR="${WATCH_MONITOR:-1}"
WATCH_INTERVAL_SEC="${WATCH_INTERVAL_SEC:-5}"
COMPLETION_NOTIFY="${COMPLETION_NOTIFY:-1}"
ALLOW_CONCURRENT_RUNS="${ALLOW_CONCURRENT_RUNS:-0}"

CONFIG_PATH="${1:-$REPO_DIR/configs/unagi_workflow.example.run.yaml}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[train-runner] missing config path: $CONFIG_PATH" >&2
  exit 2
fi

if [[ "$ALLOW_CONCURRENT_RUNS" != "1" ]]; then
  if pgrep -af "python -m UNAGI\.cli run -c ${CONFIG_PATH}" >/dev/null 2>&1; then
    echo "[train-runner] existing training run detected for config: $CONFIG_PATH" >&2
    echo "[train-runner] set ALLOW_CONCURRENT_RUNS=1 to override" >&2
    exit 5
  fi
fi

RUN_ID="unagi_training_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_DIR/outputs/$RUN_ID"
LOG_FILE="$OUT_DIR/training.log"
mkdir -p "$OUT_DIR"

MONITOR_DATA_DIR="$(python - "$CONFIG_PATH" <<'PY'
import os
import sys

cfg_path = sys.argv[1]
try:
    import yaml
except Exception:
    print("")
    raise SystemExit(0)

try:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
except Exception:
    print("")
    raise SystemExit(0)

path = (cfg.get("data") or {}).get("path", "")
if path and os.path.isfile(path):
    path = os.path.dirname(path)
print(path)
PY
)"

if [[ -z "$MONITOR_DATA_DIR" ]]; then
  MONITOR_DATA_DIR="$REPO_DIR/UNAGI/data/example"
fi

STATUS_LOG="${STATUS_LOG:-$OUT_DIR/monitor_status.log}"
mkdir -p "$(dirname "$STATUS_LOG")"
: > "$STATUS_LOG"

nohup "$CONDA_BIN" run -n "$CONDA_ENV" python -m UNAGI.cli run -c "$CONFIG_PATH" > "$LOG_FILE" 2>&1 &
PID=$!

resolve_worker_pid() {
  local parent_pid="$1"
  local child_pid=""
  local i
  for i in $(seq 1 20); do
    child_pid="$(pgrep -P "$parent_pid" | tail -n 1 || true)"
    if [[ -n "$child_pid" ]] && ps -p "$child_pid" >/dev/null 2>&1; then
      echo "$child_pid"
      return 0
    fi
    sleep 1
  done
  echo "$parent_pid"
}

MONITOR_PID="$(resolve_worker_pid "$PID")"

cat > "$OUT_DIR/run_meta.txt" <<EOF
run_id=$RUN_ID
launcher_pid=$PID
pid=$MONITOR_PID
repo_dir=$REPO_DIR
config_path=$CONFIG_PATH
monitor_data_dir=$MONITOR_DATA_DIR
log_file=$LOG_FILE
status_log=$STATUS_LOG
started_at=$(date '+%Y-%m-%d %H:%M:%S %Z')
EOF

echo "RUN_ID=$RUN_ID"
echo "OUT_DIR=$OUT_DIR"
echo "LAUNCHER_PID=$PID"
echo "PID=$MONITOR_PID"
echo "LOG_FILE=$LOG_FILE"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "MONITOR_DATA_DIR=$MONITOR_DATA_DIR"
echo "STATUS_LOG=$STATUS_LOG"

if [[ -x "$MONITOR_SCRIPT" ]] && [[ -d "$MONITOR_DATA_DIR" ]]; then
  STATUS_LOG="$STATUS_LOG" "$MONITOR_SCRIPT" "$MONITOR_DATA_DIR" "$MONITOR_PID" | tee -a "$STATUS_LOG" || true
  echo "MONITOR=started"
else
  echo "MONITOR=skipped (monitor script or data dir missing)"
fi

if [[ "$COMPLETION_NOTIFY" == "1" ]]; then
  nohup bash -lc '
    while kill -0 "'$MONITOR_PID'" 2>/dev/null; do
      sleep 60
    done
    ts="$(date "+%Y-%m-%d %H:%M:%S %Z")"
    msg="UNAGI training completed: run_id='"$RUN_ID"' at ${ts}"
    printf "%s\n" "$msg" > "'"$OUT_DIR"'/completion_notification.txt"
    command -v logger >/dev/null 2>&1 && logger -t unagi-monitor "$msg"
    command -v notify-send >/dev/null 2>&1 && notify-send "UNAGI Done" "$msg"
  ' > "$OUT_DIR/completion_watcher.log" 2>&1 &
  echo "COMPLETION_NOTIFY=started"
fi

if [[ "$WATCH_MONITOR" == "1" ]]; then
  touch "$STATUS_LOG"
  echo "WATCH=starting (interval=${WATCH_INTERVAL_SEC}s, file=$STATUS_LOG)"
  exec watch -n "$WATCH_INTERVAL_SEC" "tail -n 40 \"$STATUS_LOG\""
fi
