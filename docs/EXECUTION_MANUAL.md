# Matrix-ResGNN Execution Manual

This document is the master execution contract for the new repository. It is meant to answer four questions clearly before large-scale coding starts:

1. what will be built
2. what will be run
3. what must be saved
4. how those saved artifacts will support later paper analysis

Use this document as the source of truth for implementation and experiment organization.

## 1. Project scope

The project studies structured residual connectivity for graph classification under a unified `branch x layer` formulation.

The target paper-facing model family is:

- `Plain`
- `Vertical-Res`
- `Horizontal-Res`
- `Matrix-Res`
- `Matrix-Res (sparse/gated)`

The project will reuse the current graph-classification dataset package and parts of the training infrastructure from `cross_residual_gnn`, but the new codebase, naming, and manuscript logic must be organized around the matrix-residual framework.

## 2. Environment and top-level rule

Use the Conda environment:

```bash
conda activate pyg
```

All future Python commands in this repository should assume that environment first.

## 3. Repository responsibilities

```text
Matrix-ResGNN-GraphClassification/
├── configs/      # experiment configs
├── docs/         # planning, execution, and writing notes
├── figures/      # exported paper figures
├── paper/        # manuscript sources and method notes
├── records/      # versioned outputs, checkpoints, analysis payloads
├── scripts/      # run entrypoints and report exporters
└── src/          # reusable Python package code
```

Sub-responsibilities that must be preserved:

- `src/models/`: the five paper-facing models only
- `scripts/`: no model logic; orchestration only
- `records/LATEST/`: current paper-facing experiment outputs
- `records/LATEST/checkpoints/`: best-fold checkpoints
- `records/LATEST/analysis/`: saved states for depth, residual, and representation analysis

## 4. Code migration map from `cross_residual_gnn`

### 4.1 Files to audit first

- `geomatric/graph_classify_v3.py`
- `geomatric/experiment_catalog.py`
- `geomatric/experiment_paths.py`
- `py/run_paper_experiments.py`
- `py/export_analysis_artifacts.py`
- `py/generate_suite_analysis_figures.py`
- `py/run_residual_mode_ablation_experiments.py`
- `py/run_residual_parameter_sweeps.py`
- `py/summarize_paper_experiments.py`

### 4.2 Reuse policy

Safe to reuse after rename and cleanup:

- dataset loading
- split logic
- metric computation
- logging utilities
- batch experiment orchestration
- result summarization
- figure-export scaffolding

Must be rewritten:

- model-family definitions
- paper-facing names
- residual analysis logic
- manuscript text
- architecture figures

### 4.3 Immediate migration targets

Phase-1 migration should produce these concrete files:

- `src/experiment_catalog.py`
- `src/experiment_paths.py`
- `src/models/*.py`
- `scripts/run_benchmark.py`
- `scripts/export_residual_analysis.py`

Phase-2 migration should add:

- trainer entrypoint
- summarizer
- figure exporters
- branch-count ablation runner
- matrix sparse/gated runner

## 5. Frozen methodological assumptions

These assumptions are currently fixed unless explicitly revised:

- graph-classification task
- initial branch count `B = 3`
- later branch-count ablation `B = 1...8`
- matrix residual keeps same-layer horizontal edges
- matrix residual also includes diagonal previous-layer cross-branch edges
- sparse/gated control is a mechanism layer on top of `Matrix-Res`, not a separate topology family

## 6. Model family and experiment order

### 6.1 Main family

- `Plain`
- `VerticalRes`
- `HorizontalRes`
- `MatrixRes`
- `MatrixResGated`

### 6.2 Main benchmark order

Run order should be:

1. `Plain`
2. `VerticalRes`
3. `HorizontalRes`
4. `MatrixRes`
5. `MatrixResGated`

That order matters because:

- `Plain` checks data and trainer correctness
- `VerticalRes` checks ordinary residual reuse
- `HorizontalRes` validates branch interaction without cross-depth mixing
- `MatrixRes` is the first full topology claim
- `MatrixResGated` tests whether matrix reuse needs control

### 6.3 Main dataset order

Initial paper-facing order:

1. `PROTEINS`
2. `DD`
3. `ENZYMES`
4. `MUTAG`
5. `AIDS`
6. `Mutagenicity`

## 7. Execution phases

### Phase A. Infrastructure migration

Goal:

- move only the reusable training infrastructure
- avoid copying old paper assumptions into the new code

Required outputs:

- package import works
- dataset loading works
- split logic works
- output directories are versioned

### Phase B. Sanity runs

Goal:

- verify one dataset, one operator, one fold for all five models

Recommended minimal run:

```bash
conda activate pyg
python scripts/run_benchmark.py --dataset_group main --models Plain VerticalRes HorizontalRes MatrixRes MatrixResGated --operators GCNConv --folds 0
```

Pass condition:

- every model produces a result file
- every result file contains metrics and parameter counts
- checkpoints and history files are written

### Phase C. Main benchmark

Goal:

- reproduce the paper-facing benchmark over all datasets, folds, and operators

Expected matrix:

- datasets: 6
- models: 5
- operators: 4
- folds: 5

Total main benchmark runs:

- `6 x 5 x 4 x 5 = 600`

### Phase D. Branch-count ablation

Goal:

- test `B = 1...8` for `HorizontalRes`, `MatrixRes`, and `MatrixResGated`

Suggested scope:

- first on `PROTEINS` and `DD`
- one default operator first
- extend only after the trend is stable

### Phase E. Sparse/gated mechanism study

Goal:

- compare dense, fixed-gate, learnable-gate, top-k, and sparse shrinkage within `MatrixRes`

Suggested residual controls:

- `identity`
- `fixed_gate_0.5`
- `learnable_gate`
- `topk_0.25`
- `topk_0.50`
- `sparse_0.02`
- `sparse_0.05`

### Phase F. Representation and residual analysis

Goal:

- export the states needed for mechanism claims rather than only final accuracy

This is where the new paper will live or die. Do not skip this phase.

## 8. What must be saved for every run

Every training run must write the following minimal artifacts:

- `result.json`
- `history.json`
- `checkpoint.pt`
- `config.json`
- `train_stdout.txt` or equivalent log file

### 8.1 `result.json`

Must contain:

- dataset
- model
- operator
- fold
- seed
- branch count
- depth
- residual mode
- gate mode
- best epoch
- best validation metric
- best test metric
- loss summary
- parameter count
- runtime summary

### 8.2 `history.json`

Per epoch:

- train loss
- train accuracy
- validation loss
- validation accuracy
- learning rate
- gradient norm
- embedding norm summary

### 8.3 `checkpoint.pt`

At least:

- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict` if present
- `best_epoch`
- `best_val_metric`
- `args/config`

### 8.4 `config.json`

Must freeze:

- dataset
- operator
- model
- hidden dim
- number of layers
- number of branches
- dropout
- lr
- weight decay
- batch size
- patience
- grad clip
- gate mode
- residual mode
- sparse lambda
- top-k ratio

## 9. What must be saved for later analysis

The paper will need more than final metrics. Save these explicitly.

### 9.1 Predictions and embeddings

For paper-facing runs, save:

- `predictions.csv`
- `logits.npy`
- `labels.npy`
- `graph_embeddings.npz`

Use these for:

- confusion matrices
- calibration plots
- embedding PCA/TSNE if needed
- class-separation checks

### 9.2 Layer states

For selected representative targets, save:

- node states by branch and layer
- graph embeddings by branch and layer

Recommended file:

- `layer_states.pt`

Suggested structure:

- `node_states[branch][layer]`
- `graph_states[branch][layer]`
- `final_graph_embedding`

Do not save this for every single large run if storage becomes excessive. Restrict to representative targets.

### 9.3 Residual-path statistics

For residual analysis, save per run:

- residual source type
- residual norm before injection
- residual norm after gating or sparsification
- fraction of active channels
- mean gate value
- std of gate value

Residual source types for the new paper:

- `vertical`
- `horizontal_left`
- `horizontal_right`
- `diag_left_prev`
- `diag_right_prev`

Recommended file:

- `residual_stats.json`

### 9.4 Layer-wise gradient analysis

Save:

- gradient norm per layer
- gradient norm per branch
- deepest-layer gradient norm

Recommended file:

- `gradient_by_layer.csv`

### 9.5 Representation continuity analysis

For representative targets, save:

- pairwise CKA between adjacent layers
- pairwise CKA across branches at the same layer
- cosine similarity between branch summaries
- cosine similarity between adjacent-depth graph states

Recommended files:

- `cka_depth.csv`
- `cka_branch.csv`
- `cosine_depth.csv`
- `cosine_branch.csv`

### 9.6 Branch diversity analysis

Because the new paper explicitly uses multiple branches, save:

- branch embedding pairwise distance
- branch variance contribution
- branch collapse indicators

Recommended file:

- `branch_diversity.json`

## 10. Which analyses the paper is expected to contain

These are not optional if the paper wants to make mechanism claims.

### 10.1 Main benchmark analysis

- mean accuracy by dataset and operator
- rank table across models
- parameter count
- best epoch summary

### 10.2 Residual topology analysis

- `VerticalRes - Plain`
- `HorizontalRes - Plain`
- `MatrixRes - max(VerticalRes, HorizontalRes)`
- `MatrixResGated - MatrixRes`

This is the new equivalent of the old cross-advantage story.

### 10.3 Branch-count analysis

- performance as a function of `B`
- parameter growth vs performance gain
- runtime and memory vs `B`

### 10.4 Residual-control analysis

- dense vs learnable-gate
- dense vs sparse
- top-k vs sparse
- active-channel ratio vs accuracy

### 10.5 Representation-dynamics analysis

- continuity vs performance
- branch diversity vs performance
- gradient stability vs performance
- whether matrix reuse causes useful diversification or branch collapse

## 11. Recommended file layout under `records/LATEST/`

```text
records/LATEST/
├── benchmark/
│   ├── json/
│   ├── csv/
│   └── summaries/
├── checkpoints/
├── logs/
├── runs/
├── analysis/
│   ├── representative_targets/
│   ├── residual_stats/
│   ├── cka/
│   ├── cosine/
│   └── gradients/
└── manifests/
```

## 12. Naming convention

All per-run filenames should include at least:

- dataset
- model
- operator
- fold
- branch count if not default
- residual control if not default

Example stem:

```text
PROTEINS__MatrixRes__GCNConv__fold0__B3__identity
```

## 13. Planned commands

These are the commands the repository should support after migration.

### 13.1 Single run

```bash
conda activate pyg
python scripts/run_single.py --dataset PROTEINS --model MatrixRes --operator GCNConv --fold 0
```

### 13.2 Main benchmark

```bash
conda activate pyg
python scripts/run_benchmark.py --dataset_group all --folds 0 1 2 3 4
```

### 13.3 Branch-count ablation

```bash
conda activate pyg
python scripts/run_branch_ablation.py --dataset PROTEINS --model MatrixRes --operator GCNConv --branches 1 2 3 4 5 6 7 8
```

### 13.4 Residual-control ablation

```bash
conda activate pyg
python scripts/run_matrix_control_ablation.py --dataset DD --operator GCNConv
```

### 13.5 Analysis export

```bash
conda activate pyg
python scripts/export_residual_analysis.py --dataset PROTEINS --model MatrixRes --operator GCNConv --fold 0
```

### 13.6 Benchmark summarization and figure export

```bash
conda activate pyg
python scripts/summarize_benchmark.py
python scripts/generate_suite_figures.py
```

## 14. Implementation priority order

Build in this order:

1. migrate trainer
2. migrate dataset loading and split logic
3. implement `Plain`
4. implement `VerticalRes`
5. implement `HorizontalRes`
6. implement `MatrixRes`
7. implement `MatrixResGated`
8. add benchmark runner
9. add summarizer
10. add analysis exporter
11. add branch ablation runner
12. add figure/table exporters

## 15. Decision checkpoints

Before scaling experiments, stop and confirm these checkpoints:

### Checkpoint 1

- one dataset
- one operator
- five models
- no crashes

### Checkpoint 2

- one full dataset package
- stable output files
- summarizer works

### Checkpoint 3

- representative residual-analysis payloads are usable
- branch and layer states are saved correctly
- CKA and cosine computations are reproducible

### Checkpoint 4

- branch-count ablation gives interpretable trends
- matrix sparse/gated control yields a meaningful mechanism story

## 16. Non-negotiable rule

Do not run large benchmark batches until the repository can already:

- save checkpoints
- save per-epoch history
- save representative layer states
- save residual-path statistics
- export a compact benchmark summary

Otherwise the paper will end up with final accuracies but no mechanism evidence.
