#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-unagi}"
MONITOR_SCRIPT="$REPO_DIR/scripts/monitor_unagi_passive.sh"
MONITOR_DATA_DIR="$REPO_DIR/UNAGI/data/example"
WATCH_MONITOR="${WATCH_MONITOR:-1}"
WATCH_INTERVAL_SEC="${WATCH_INTERVAL_SEC:-5}"
COMPLETION_NOTIFY="${COMPLETION_NOTIFY:-1}"

ITERATION="${1:-2}"
SAMPLING_TIMES="${2:-20}"
DATA_PATH="$REPO_DIR/UNAGI/data/example/${ITERATION}/stagedata/dataset.h5ad"
DRUG_DB="$REPO_DIR/UNAGI/data/cmap_drug_target_genes.npy"

if [[ ! -f "$DATA_PATH" ]]; then
  echo "[post-runner] missing data path: $DATA_PATH" >&2
  exit 2
fi

if [[ ! -f "$DRUG_DB" ]]; then
  echo "[post-runner] missing drug DB: $DRUG_DB" >&2
  exit 2
fi

RUN_ID="post_analysis_iter${ITERATION}_full_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$REPO_DIR/outputs/$RUN_ID"
LOG_FILE="$OUT_DIR/post_analysis.log"
PY_FILE="$OUT_DIR/run_post_analysis.py"
mkdir -p "$OUT_DIR"
STATUS_LOG="${STATUS_LOG:-$OUT_DIR/monitor_status.log}"
mkdir -p "$(dirname "$STATUS_LOG")"
: > "$STATUS_LOG"

cat > "$PY_FILE" <<'PY'
import os
import sys
import traceback
import warnings

warnings.filterwarnings(
  "ignore",
  category=FutureWarning,
  module=r".*hierachical_static_markers.*",
)
warnings.filterwarnings(
  "ignore",
  category=UserWarning,
  module=r"anndata\.compat\._overloaded_dict",
)
warnings.filterwarnings(
  "ignore",
  message=r"Trying to modify attribute .* of view, initializing view as actual\.",
)

repo_dir = os.environ["UNAGI_REPO_DIR"]
out_dir = sys.argv[1]
iteration = int(sys.argv[2])
sampling_times = int(sys.argv[3])

sys.path.insert(0, repo_dir)
import UNAGI as UNAGI_pkg
from UNAGI.UNAGI_tool import UNAGI

print(f"[POST] module={UNAGI_pkg.__file__}", flush=True)
print(f"[POST] out={out_dir}", flush=True)
print(f"[POST] iteration={iteration} sampling_times={sampling_times} centroid=False", flush=True)

try:
    unagi = UNAGI()
    unagi.analyse_UNAGI(
        data_path=f"{repo_dir}/UNAGI/data/example/{iteration}/stagedata/dataset.h5ad",
        iteration=iteration,
        random_background_sampling_times=sampling_times,
        run_perturbation=True,
        customized_drug=f"{repo_dir}/UNAGI/data/cmap_drug_target_genes.npy",
        target_dir=out_dir,
        centroid=False,
    )
    print("[POST] DONE", flush=True)
except Exception as exc:
    print(f"[POST] ERROR: {exc}", flush=True)
    traceback.print_exc()
    raise
PY

nohup env UNAGI_REPO_DIR="$REPO_DIR" "$CONDA_BIN" run -n "$CONDA_ENV" python -u "$PY_FILE" "$OUT_DIR" "$ITERATION" "$SAMPLING_TIMES" > "$LOG_FILE" 2>&1 &
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
iteration=$ITERATION
sampling_times=$SAMPLING_TIMES
centroid=false
data_path=$DATA_PATH
drug_db=$DRUG_DB
log_file=$LOG_FILE
status_log=$STATUS_LOG
started_at=$(date '+%Y-%m-%d %H:%M:%S %Z')
EOF

echo "RUN_ID=$RUN_ID"
echo "OUT_DIR=$OUT_DIR"
echo "LAUNCHER_PID=$PID"
echo "PID=$MONITOR_PID"
echo "LOG_FILE=$LOG_FILE"
echo "STATUS_LOG=$STATUS_LOG"

if [[ -x "$MONITOR_SCRIPT" ]]; then
  STATUS_LOG="$STATUS_LOG" "$MONITOR_SCRIPT" "$MONITOR_DATA_DIR" "$MONITOR_PID" | tee -a "$STATUS_LOG" || true
  echo "MONITOR=started"
else
  echo "MONITOR=skipped (missing executable: $MONITOR_SCRIPT)"
fi

if [[ "$COMPLETION_NOTIFY" == "1" ]]; then
  nohup bash -lc '
    while kill -0 "'$MONITOR_PID'" 2>/dev/null; do
      sleep 60
    done
    ts="$(date "+%Y-%m-%d %H:%M:%S %Z")"
    msg="UNAGI post-analysis completed: run_id='"$RUN_ID"' at ${ts}"
    printf "%s\n" "$msg" > "'"$OUT_DIR"'/completion_notification.txt"
    command -v logger >/dev/null 2>&1 && logger -t unagi-monitor "$msg"
    command -v notify-send >/dev/null 2>&1 && notify-send "UNAGI Post Done" "$msg"
  ' > "$OUT_DIR/completion_watcher.log" 2>&1 &
  echo "COMPLETION_NOTIFY=started"
fi

if [[ "$WATCH_MONITOR" == "1" ]]; then
  touch "$STATUS_LOG"
  echo "WATCH=starting (interval=${WATCH_INTERVAL_SEC}s, file=$STATUS_LOG)"
  exec watch -n "$WATCH_INTERVAL_SEC" "tail -n 40 \"$STATUS_LOG\""
fi
