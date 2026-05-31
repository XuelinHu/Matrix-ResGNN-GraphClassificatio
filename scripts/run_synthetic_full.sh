#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
VERSION="${VERSION:-SYN_FULL}"
PROFILE="${PROFILE:-paper}"
QUEUE_NAME="${QUEUE_NAME:-synthetic_full_5dist_4ops_5folds}"
MAX_WORKERS="${MAX_WORKERS:-4}"

DATASETS=(SYN_ER SYN_BA SYN_SBM SYN_WS SYN_REGULAR)
MODELS=(Plain VerticalRes HorizontalRes MatrixRes MatrixResGated)
OPERATORS=(GCNConv GATConv SAGEConv GINConv)
FOLDS=(0 1 2 3 4)

"$PYTHON_BIN" scripts/generate_synthetic_graphs.py \
  --profile "$PROFILE" \
  --datasets "${DATASETS[@]}" \
  --graphs_per_class "${GRAPHS_PER_CLASS:-200}" \
  --min_nodes "${MIN_NODES:-30}" \
  --max_nodes "${MAX_NODES:-80}" \
  --feature_dim "${FEATURE_DIM:-8}" \
  --force

"$PYTHON_BIN" scripts/run_missing_benchmark_queue.py \
  --version "$VERSION" \
  --synthetic_profile "$PROFILE" \
  --datasets "${DATASETS[@]}" \
  --models "${MODELS[@]}" \
  --operators "${OPERATORS[@]}" \
  --folds "${FOLDS[@]}" \
  --max_workers "$MAX_WORKERS" \
  --queue_name "$QUEUE_NAME"

"$PYTHON_BIN" scripts/summarize_benchmark.py \
  --version "$VERSION" \
  --datasets "${DATASETS[@]}" \
  --models "${MODELS[@]}" \
  --operators "${OPERATORS[@]}"
