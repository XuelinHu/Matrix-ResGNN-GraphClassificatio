# Experiment Stage Manifest

Date:

- Beijing time: `2026-05-07`

Scope:

1. branch-count ablation
2. first-batch parameter sensitivity scans

Branch-count ablation targets:

- datasets: `PROTEINS`, `DD`
- models:
  - `HorizontalRes`
  - `MatrixRes`
  - `MatrixResGated`
- operator: `GCNConv`
- branch counts: `1..8`
- folds: `0..4`

First-batch parameter sensitivity targets:

- datasets: `PROTEINS`, `DD`
- models:
  - `MatrixRes`
  - `MatrixResGated`
- operator: `GCNConv`
- folds: `0`
- sweeps:
  - `lr`
  - `drop`
  - `dim`
  - `sparse_lambda`
  - `gate_init`

Execution notes:

- branch ablation and sensitivity scans are allowed to run in parallel
- preserve all outputs under `records/LATEST/`
- avoid mutating the training core while long jobs are running
