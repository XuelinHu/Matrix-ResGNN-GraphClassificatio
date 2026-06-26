# Experiment Stage Manifest

Start time:

- Beijing time: `2026-05-06` night session

Execution policy:

- do not change the paper framing during the run stage
- prefer fixing implementation issues locally rather than pausing the run stage
- preserve all outputs under `records/LATEST/`

Current run order:

1. main benchmark on `PROTEINS`, `DD`, `ENZYMES`
2. operator: `GCNConv`
3. models:
   - `Plain`
   - `VerticalRes`
   - `HorizontalRes`
   - `MatrixRes`
   - `MatrixResGated`
4. folds: `0,1,2,3,4`

Planned follow-up after the first benchmark finishes:

1. regenerate benchmark summary CSV
2. regenerate suite figures
3. inspect failures or missing folds if any
4. if the main benchmark is stable, continue with branch-count ablation on `PROTEINS` and `DD`

Primary records to preserve:

- `records/LATEST/logs/`
- `records/LATEST/checkpoints/`
- `records/LATEST/analysis/`
- `records/LATEST/summaries/`
