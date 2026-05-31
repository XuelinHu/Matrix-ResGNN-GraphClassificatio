#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEFAULT_PYTHON="$(command -v python)"
if [[ -x "/home/xuelin/miniconda3/envs/pyg/bin/python" ]]; then
  DEFAULT_PYTHON="/home/xuelin/miniconda3/envs/pyg/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
VERSION="${VERSION:-SYN_MULTI_FULL}"
PROFILE="${PROFILE:-paper}"
QUEUE_NAME="${QUEUE_NAME:-synthetic_multiclass_full_5dist_c2c8_4ops_5folds}"
MAX_WORKERS="${MAX_WORKERS:-4}"

BASE_DATASETS=(SYN_ER SYN_BA SYN_SBM SYN_WS SYN_REGULAR)
CLASS_COUNTS=(2 3 4 5 6 7 8)
DATASETS=()
for dataset in "${BASE_DATASETS[@]}"; do
  for class_count in "${CLASS_COUNTS[@]}"; do
    DATASETS+=("${dataset}_C${class_count}")
  done
done

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
