#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEFAULT_PYTHON="$(command -v python)"
if [[ -x "/home/xuelin/miniconda3/envs/pyg/bin/python" ]]; then
  DEFAULT_PYTHON="/home/xuelin/miniconda3/envs/pyg/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
VERSION="${VERSION:-SYN_MULTI_SMOKE}"
PROFILE="${PROFILE:-smoke}"
QUEUE_NAME="${QUEUE_NAME:-synthetic_multiclass_smoke_5dist_c2c8_gcn_fold0}"
MAX_WORKERS="${MAX_WORKERS:-4}"
OVERRIDE_EP="${OVERRIDE_EP:-3}"
OVERRIDE_DIM="${OVERRIDE_DIM:-16}"
OVERRIDE_H_LAYER="${OVERRIDE_H_LAYER:-2}"
OVERRIDE_PATIENCE="${OVERRIDE_PATIENCE:-2}"
OVERRIDE_BATCH_SIZE="${OVERRIDE_BATCH_SIZE:-32}"

BASE_DATASETS=(SYN_ER SYN_BA SYN_SBM SYN_WS SYN_REGULAR)
CLASS_COUNTS=(2 3 4 5 6 7 8)
DATASETS=()
for dataset in "${BASE_DATASETS[@]}"; do
  for class_count in "${CLASS_COUNTS[@]}"; do
    DATASETS+=("${dataset}_C${class_count}")
  done
done

MODELS=(Plain VerticalRes HorizontalRes MatrixRes MatrixResGated)

"$PYTHON_BIN" scripts/generate_synthetic_graphs.py \
  --profile "$PROFILE" \
  --datasets "${DATASETS[@]}" \
  --graphs_per_class 12 \
  --min_nodes 16 \
  --max_nodes 28 \
  --feature_dim 8 \
  --force

"$PYTHON_BIN" scripts/run_missing_benchmark_queue.py \
  --version "$VERSION" \
  --synthetic_profile "$PROFILE" \
  --datasets "${DATASETS[@]}" \
  --models "${MODELS[@]}" \
  --operators GCNConv \
  --folds 0 \
  --max_workers "$MAX_WORKERS" \
  --queue_name "$QUEUE_NAME" \
  --override_ep "$OVERRIDE_EP" \
  --override_dim "$OVERRIDE_DIM" \
  --override_h_layer "$OVERRIDE_H_LAYER" \
  --override_patience "$OVERRIDE_PATIENCE" \
  --override_batch_size "$OVERRIDE_BATCH_SIZE" \
  --force_rerun

"$PYTHON_BIN" scripts/summarize_benchmark.py \
  --version "$VERSION" \
  --datasets "${DATASETS[@]}" \
  --models "${MODELS[@]}" \
  --operators GCNConv
