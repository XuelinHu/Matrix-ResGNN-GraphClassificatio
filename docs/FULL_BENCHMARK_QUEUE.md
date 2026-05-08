# Full Benchmark Queue

This task tracks the expanded benchmark requested to match the reference-paper coverage while keeping the current Matrix-ResGNN paper framing.

## Scope

Datasets:

- PROTEINS
- DD
- ENZYMES
- MUTAG
- AIDS
- Mutagenicity

Models:

- Plain
- VerticalRes
- HorizontalRes
- MatrixRes
- MatrixResGated

Operators:

- GCNConv
- GATConv
- SAGEConv
- GINConv

Folds:

- 0, 1, 2, 3, 4

Total benchmark size:

- 6 datasets x 5 models x 4 operators x 5 folds = 600 jobs

Initial estimate before queue launch:

- Main GCNConv results: 75 jobs
- Supplementary MUTAG GCNConv results from the interrupted preliminary run: 25 jobs
- Estimated missing: 500 jobs

Actual dry-run status at queue launch:

- Expected total: 600 jobs
- Pending at start: 425 jobs
- The difference comes from additional jobs that completed during the interrupted preliminary run before the queue was formalized.

## Runner

Use the resumable queue runner:

```bash
conda run -n pyg python scripts/run_missing_benchmark_queue.py \
  --datasets PROTEINS DD ENZYMES MUTAG AIDS Mutagenicity \
  --models Plain VerticalRes HorizontalRes MatrixRes MatrixResGated \
  --operators GCNConv GATConv SAGEConv GINConv \
  --folds 0 1 2 3 4 \
  --max_workers 4
```

The runner skips already completed result keys under `records/LATEST/logs/`.

## Background Launch

Recommended non-blocking launch. `setsid` is used so the job survives after the interactive shell turn ends:

```bash
setsid /home/xuelin/miniconda3/envs/pyg/bin/python scripts/run_missing_benchmark_queue.py \
  --datasets PROTEINS DD ENZYMES MUTAG AIDS Mutagenicity \
  --models Plain VerticalRes HorizontalRes MatrixRes MatrixResGated \
  --operators GCNConv GATConv SAGEConv GINConv \
  --folds 0 1 2 3 4 \
  --max_workers 4 \
  > records/LATEST/queue/full_benchmark_6datasets_4ops.nohup.log 2>&1 < /dev/null &
```

## Progress Files

The runner writes:

- `records/LATEST/queue/full_benchmark_6datasets_4ops_status.json`
- `records/LATEST/queue/full_benchmark_6datasets_4ops_events.jsonl`
- `records/LATEST/queue/full_benchmark_6datasets_4ops_plan.json`
- `records/LATEST/queue/full_benchmark_6datasets_4ops.nohup.log`

Check progress:

```bash
cat records/LATEST/queue/full_benchmark_6datasets_4ops_status.json
tail -n 20 records/LATEST/queue/full_benchmark_6datasets_4ops.nohup.log
```

Check completeness:

```bash
python scripts/check_benchmark_completeness.py \
  --datasets PROTEINS DD ENZYMES MUTAG AIDS Mutagenicity \
  --models Plain VerticalRes HorizontalRes MatrixRes MatrixResGated \
  --operators GCNConv GATConv SAGEConv GINConv \
  --folds 0 1 2 3 4
```

## After Completion

Run summaries and regenerate figures:

```bash
conda run -n pyg python scripts/summarize_benchmark.py \
  --datasets PROTEINS DD ENZYMES MUTAG AIDS Mutagenicity \
  --models Plain VerticalRes HorizontalRes MatrixRes MatrixResGated \
  --operators GCNConv GATConv SAGEConv GINConv

conda run -n pyg python scripts/generate_suite_figures.py
```

Then update the PeerJ manuscript:

- Main text: retain PROTEINS, DD, and ENZYMES as the primary benchmark.
- Supplementary robustness: add MUTAG, AIDS, and Mutagenicity.
- Add operator-level table or appendix table covering GCNConv, GATConv, SAGEConv, and GINConv.
- Add winner/rank summary across dataset-operator combinations.
