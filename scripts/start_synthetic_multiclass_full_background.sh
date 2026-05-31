#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERSION="${VERSION:-SYN_MULTI_FULL}"
QUEUE_DIR="records/${VERSION}/queue"
LOG_PATH="${QUEUE_DIR}/synthetic_multiclass_full.nohup.log"

mkdir -p "$QUEUE_DIR"
setsid bash scripts/run_synthetic_multiclass_full.sh > "$LOG_PATH" 2>&1 < /dev/null &
PID="$!"

echo "$PID" > "${QUEUE_DIR}/synthetic_multiclass_full.pid"
echo "Synthetic multi-class full benchmark started"
echo "PID: $PID"
echo "Log: $LOG_PATH"
