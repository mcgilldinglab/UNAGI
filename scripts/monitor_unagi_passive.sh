#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: monitor_unagi_passive.sh <data_dir> [pid]

Env vars:
  USE_CRON          Set to 1 to install/use cron by default (default: 1)
  CRON_INTERVAL_MIN Cron interval in minutes (default: 15)
  CRON_MODE         Internal flag used by cron invocation (default: 0)
  ONCE              Set to 1 to print one snapshot and exit (default: 0)
  INTERVAL_SEC      Poll interval in seconds for watch mode (default: 60)
  STATUS_LOG        Log file for cron snapshots (default: <data_dir>/monitor_status.log)
  STATUS_EMAIL      Optional email recipient for each snapshot (requires mail command)
  STATUS_WEBHOOK_URL Optional webhook URL to POST plaintext status

Examples:
  ONCE=1 monitor_unagi_passive.sh ./UNAGI/data/example 2258478
  USE_CRON=1 monitor_unagi_passive.sh ./UNAGI/data/example 2258478
  USE_CRON=0 INTERVAL_SEC=90 monitor_unagi_passive.sh ./UNAGI/data/example 2258478
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

DATA_DIR="$1"
PID="${2:-}"
USE_CRON="${USE_CRON:-1}"
CRON_INTERVAL_MIN="${CRON_INTERVAL_MIN:-15}"
CRON_MODE="${CRON_MODE:-0}"
INTERVAL_SEC="${INTERVAL_SEC:-60}"
ONCE="${ONCE:-0}"
STATUS_LOG="${STATUS_LOG:-$DATA_DIR/monitor_status.log}"
STATUS_EMAIL="${STATUS_EMAIL:-}"
STATUS_WEBHOOK_URL="${STATUS_WEBHOOK_URL:-}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[monitor] data directory not found: $DATA_DIR" >&2
  exit 2
fi

if [[ ! "$CRON_INTERVAL_MIN" =~ ^[0-9]+$ ]] || [[ "$CRON_INTERVAL_MIN" -lt 1 ]] || [[ "$CRON_INTERVAL_MIN" -gt 59 ]]; then
  echo "[monitor] CRON_INTERVAL_MIN must be an integer between 1 and 59 (got: $CRON_INTERVAL_MIN)" >&2
  exit 3
fi

if [[ ! "$INTERVAL_SEC" =~ ^[0-9]+$ ]] || [[ "$INTERVAL_SEC" -lt 1 ]]; then
  echo "[monitor] INTERVAL_SEC must be a positive integer (got: $INTERVAL_SEC)" >&2
  exit 4
fi

script_path() {
  if command -v readlink >/dev/null 2>&1; then
    readlink -f "$0" 2>/dev/null || echo "$0"
  else
    echo "$0"
  fi
}

shell_quote() {
  printf '%q' "$1"
}

cron_id() {
  local key digest
  key="${DATA_DIR}|${PID}"
  if command -v sha1sum >/dev/null 2>&1; then
    digest="$(printf '%s' "$key" | sha1sum | awk '{print $1}')"
  elif command -v md5sum >/dev/null 2>&1; then
    digest="$(printf '%s' "$key" | md5sum | awk '{print $1}')"
  else
    digest="$(printf '%s' "$key" | cksum | awk '{print $1}')"
  fi
  echo "UNAGI_MONITOR_${digest}"
}

CRON_TAG="$(cron_id)"

get_latest_iter_dir() {
  local latest_iter
  latest_iter="$( (find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null || true) | grep -E '^[0-9]+$' | sort -n | tail -n 1 )"
  if [[ -n "$latest_iter" ]]; then
    echo "$DATA_DIR/$latest_iter"
  fi
}

is_unagi_finished() {
  local latest_iter_dir=""
  local latest_complete=0

  latest_iter_dir="$(get_latest_iter_dir || true)"
  if [[ -n "$latest_iter_dir" ]]; then
    local has_edges has_avg idrem_count
    has_edges=0
    has_avg=0
    idrem_count=0

    [[ -f "$latest_iter_dir/edges.txt" ]] && has_edges=1
    [[ -f "$latest_iter_dir/averageValues.npy" ]] && has_avg=1
    if [[ -d "$latest_iter_dir/idremResults" ]]; then
      idrem_count="$(find "$latest_iter_dir/idremResults" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
    fi

    if [[ "$has_edges" -eq 1 ]] && [[ "$has_avg" -eq 1 ]] && [[ "$idrem_count" -gt 0 ]]; then
      latest_complete=1
    fi
  fi

  if [[ -n "$PID" ]]; then
    if ps -p "$PID" >/dev/null 2>&1; then
      return 1
    fi
    return 0
  fi

  if [[ "$latest_complete" -eq 1 ]]; then
    return 0
  fi

  return 1
}

remove_cron_entry() {
  if ! command -v crontab >/dev/null 2>&1; then
    return 0
  fi

  local current filtered
  current="$(crontab -l 2>/dev/null || true)"
  filtered="$(printf '%s\n' "$current" | grep -F -v "$CRON_TAG" || true)"
  if [[ "$current" != "$filtered" ]]; then
    printf '%s\n' "$filtered" | crontab -
    echo "[monitor] removed cron task tag=$CRON_TAG"
  fi
}

install_cron_entry() {
  if ! command -v crontab >/dev/null 2>&1; then
    echo "[monitor] crontab command not found; cannot install cron task" >&2
    return 1
  fi

  local current cron_line script
  script="$(script_path)"
  current="$(crontab -l 2>/dev/null || true)"

  if printf '%s\n' "$current" | grep -F -q "$CRON_TAG"; then
    echo "[monitor] cron task already exists tag=$CRON_TAG"
    return 0
  fi

  cron_line="*/${CRON_INTERVAL_MIN} * * * * ONCE=1 CRON_MODE=1 USE_CRON=1 STATUS_LOG=$(shell_quote "$STATUS_LOG") STATUS_EMAIL=$(shell_quote "$STATUS_EMAIL") STATUS_WEBHOOK_URL=$(shell_quote "$STATUS_WEBHOOK_URL") $(shell_quote "$script") $(shell_quote "$DATA_DIR") $(shell_quote "$PID") >> $(shell_quote "$STATUS_LOG") 2>&1 # ${CRON_TAG}"

  {
    printf '%s\n' "$current"
    printf '%s\n' "$cron_line"
  } | crontab -

  echo "[monitor] installed cron task every ${CRON_INTERVAL_MIN} min tag=$CRON_TAG"
}

send_status() {
  local snapshot="$1"

  if [[ -n "$STATUS_EMAIL" ]] && command -v mail >/dev/null 2>&1; then
    printf '%s\n' "$snapshot" | mail -s "UNAGI status: $(basename "$DATA_DIR")" "$STATUS_EMAIL" || true
  fi

  if [[ -n "$STATUS_WEBHOOK_URL" ]] && command -v curl >/dev/null 2>&1; then
    curl -fsS -X POST -H 'Content-Type: text/plain' --data-binary "$snapshot" "$STATUS_WEBHOOK_URL" >/dev/null || true
  fi

  if command -v logger >/dev/null 2>&1; then
    logger -t unagi-monitor "snapshot data_dir=$DATA_DIR pid=${PID:-N/A}"
  fi
}

print_snapshot() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "============================================================"
  echo "[monitor] $ts"
  echo "[monitor] data_dir=$DATA_DIR"

  if [[ -n "$PID" ]]; then
    if ps -p "$PID" >/dev/null 2>&1; then
      local ps_line
      ps_line="$(ps -p "$PID" -o pid=,etime=,%cpu=,%mem=,stat=,cmd=)"
      echo "[monitor] process=RUNNING $ps_line"
    else
      echo "[monitor] process=NOT_RUNNING pid=$PID"
    fi
  fi

  if [[ -d "$DATA_DIR/model_save" ]]; then
    local ckpt_count dis_ckpt_count
    ckpt_count="$(find "$DATA_DIR/model_save" -maxdepth 1 -type f -name '*_[0-9]*.pth' ! -name '*_dis_*' | wc -l | tr -d ' ')"
    dis_ckpt_count="$(find "$DATA_DIR/model_save" -maxdepth 1 -type f -name '*_dis_[0-9]*.pth' | wc -l | tr -d ' ')"
    echo "[monitor] checkpoints=model:${ckpt_count} discriminator:${dis_ckpt_count}"
  else
    echo "[monitor] checkpoints=model_save_missing"
  fi

  local found_iter=0
  local iter_dir
  for iter_dir in "$DATA_DIR"/*; do
    if [[ ! -d "$iter_dir" ]]; then
      continue
    fi
    local iter_id
    iter_id="$(basename "$iter_dir")"
    if [[ ! "$iter_id" =~ ^[0-9]+$ ]]; then
      continue
    fi
    found_iter=1
    local stage_count idrem_count has_edges has_avg loss_line

    if [[ -d "$iter_dir/stagedata" ]]; then
      stage_count="$(find "$iter_dir/stagedata" -maxdepth 1 -type f -name '[0-9]*.h5ad' | wc -l | tr -d ' ')"
    else
      stage_count=0
    fi

    if [[ -d "$iter_dir/idremResults" ]]; then
      idrem_count="$(find "$iter_dir/idremResults" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
    else
      idrem_count=0
    fi

    if [[ -f "$iter_dir/edges.txt" ]]; then
      has_edges=yes
    else
      has_edges=no
    fi

    if [[ -f "$iter_dir/averageValues.npy" ]]; then
      has_avg=yes
    else
      has_avg=no
    fi

    if [[ -f "$iter_dir/loss.txt" ]]; then
      loss_line="$(tail -n 1 "$iter_dir/loss.txt" | tr -d '\n')"
    else
      loss_line="N/A"
    fi

    echo "[monitor] iter=${iter_id} stagedata_h5ad=${stage_count} edges=${has_edges} averageValues=${has_avg} idrem_tracks=${idrem_count}"
    echo "[monitor] iter=${iter_id} last_loss=${loss_line}"
  done

  if [[ "$found_iter" -eq 0 ]]; then
    echo "[monitor] no iteration directories found yet (expected: 0,1,2,...)"
  fi

  if is_unagi_finished; then
    echo "[monitor] overall_status=FINISHED"
  else
    echo "[monitor] overall_status=RUNNING"
  fi
}

if [[ "$ONCE" == "1" ]]; then
  snapshot="$(print_snapshot)"
  printf '%s\n' "$snapshot"
  send_status "$snapshot"

  if is_unagi_finished; then
    remove_cron_entry
  fi

  exit 0
fi

if [[ "$USE_CRON" == "1" ]] && [[ "$CRON_MODE" == "0" ]]; then
  install_cron_entry
  snapshot="$(print_snapshot)"
  printf '%s\n' "$snapshot"
  send_status "$snapshot"

  if is_unagi_finished; then
    remove_cron_entry
  fi

  exit 0
fi

while true; do
  snapshot="$(print_snapshot)"
  printf '%s\n' "$snapshot"
  send_status "$snapshot"
  if is_unagi_finished; then
    remove_cron_entry
    exit 0
  fi
  sleep "$INTERVAL_SEC"
done
