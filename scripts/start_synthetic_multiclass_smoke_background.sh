#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERSION="${VERSION:-SYN_MULTI_SMOKE}"
QUEUE_DIR="records/${VERSION}/queue"
LOG_PATH="${QUEUE_DIR}/synthetic_multiclass_smoke.nohup.log"

mkdir -p "$QUEUE_DIR"
setsid bash scripts/run_synthetic_multiclass_smoke.sh > "$LOG_PATH" 2>&1 < /dev/null &
PID="$!"

echo "$PID" > "${QUEUE_DIR}/synthetic_multiclass_smoke.pid"
echo "Synthetic multi-class smoke benchmark started"
echo "PID: $PID"
echo "Log: $LOG_PATH"
