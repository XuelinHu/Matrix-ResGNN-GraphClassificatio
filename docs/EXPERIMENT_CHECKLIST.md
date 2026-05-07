# Matrix-ResGNN Experiment Checklist

This document is the paper-facing experiment checklist for the new repository. It is intended to answer three questions together:

1. which experiments must be run
2. which intermediate artifacts must be saved
3. how those artifacts will support later interpretation and writing

Use this file as the execution index for the project.

## 1. Scope

The paper studies structured residual connectivity for graph classification under a unified `branch x layer` view.

Main model family:

- `Plain`
- `VerticalRes`
- `HorizontalRes`
- `MatrixRes`
- `MatrixResGated`

Main datasets:

- `PROTEINS`
- `DD`
- `ENZYMES`

Supplementary datasets:

- `MUTAG`
- `AIDS`
- `Mutagenicity`

Main operators:

- `GCNConv`
- `GATConv`
- `SAGEConv`
- `GINConv`

## 2. Experiment hierarchy

The full experiment plan is divided into six layers:

1. main benchmark
2. structure ablation
3. branch-count ablation
4. residual-control ablation
5. parameter sensitivity
6. mechanism and representation analysis

These layers are not interchangeable. The first four establish performance facts; the last two explain them.

## 3. Main benchmark

### 3.1 Goal

Establish the baseline performance matrix under a shared protocol.

### 3.2 Matrix

- datasets: `PROTEINS`, `DD`, `ENZYMES`, `MUTAG`, `AIDS`, `Mutagenicity`
- models: `Plain`, `VerticalRes`, `HorizontalRes`, `MatrixRes`, `MatrixResGated`
- operators: `GCNConv`, `GATConv`, `SAGEConv`, `GINConv`
- folds: `0..4`

### 3.3 Minimum paper-facing order

First complete:

- `PROTEINS`, `DD`, `ENZYMES`
- all 5 models
- `GCNConv`
- all 5 folds

Then expand to:

- supplementary datasets
- the other 3 operators

### 3.4 Outputs

- fold-level JSON logs
- benchmark summary CSV
- main bar charts
- model win-count chart

## 4. Structure ablation

### 4.1 Goal

Answer whether matrix-style residual reuse is stronger than one-axis residual reuse.

### 4.2 Required comparisons

- `Plain` vs `VerticalRes`
- `Plain` vs `HorizontalRes`
- `Plain` vs `MatrixRes`
- `MatrixRes` vs `max(VerticalRes, HorizontalRes)`
- `MatrixResGated` vs `MatrixRes`

### 4.3 Interpretation target

This section should support statements such as:

- vertical reuse alone helps or does not help
- horizontal exchange alone helps or does not help
- matrix-style local 2D reuse adds value beyond either axis alone
- matrix residual benefits from gated or sparse control, or does not

### 4.4 Required outputs

- pairwise delta table
- per-dataset winner table
- fold-level comparison rows

## 5. Branch-count ablation

### 5.1 Goal

Measure how the number of branches changes performance, efficiency, and representation diversity.

### 5.2 Models

- `HorizontalRes`
- `MatrixRes`
- `MatrixResGated`

### 5.3 Branch counts

- `B = 1, 2, 3, 4, 5, 6, 7, 8`

### 5.4 Suggested order

Run first on:

- `PROTEINS`
- `DD`
- `GCNConv`

Then expand if the trend is interpretable.

### 5.5 What to report

- accuracy vs `B`
- runtime vs `B`
- parameter count vs `B`
- branch diversity vs `B`
- branch collapse indicators vs `B`

### 5.6 Interpretation target

This section should answer:

- does more branching actually help
- is there a useful range for `B`
- when does extra branching become redundant or unstable

## 6. Residual-control ablation

### 6.1 Goal

Test whether matrix residuals should be injected densely or selectively.

### 6.2 Main model

- `MatrixResGated`

Optional baseline:

- `MatrixRes`

### 6.3 Control modes

- `identity`
- `fixed_gate_0.5`
- `learnable_gate`
- `topk_0.25`
- `topk_0.50`
- `sparse_0.02`
- `sparse_0.05`

### 6.4 Suggested order

Run first on:

- `PROTEINS`
- `DD`
- `GCNConv`

### 6.5 What to report

- mean/std accuracy
- active channel ratio
- gate value summary
- residual norm before and after filtering
- runtime impact

### 6.6 Interpretation target

This section should answer:

- whether dense matrix reuse is too noisy
- whether sparse/gated control improves stability
- whether the best control is dataset-dependent

## 7. Parameter sensitivity

### 7.1 Goal

Evaluate whether the new structure is robust or only works in a narrow hyperparameter region.

### 7.2 Primary models

- `MatrixRes`
- `MatrixResGated`

Secondary models if needed:

- `VerticalRes`
- `HorizontalRes`

### 7.3 Primary datasets

- `PROTEINS`
- `DD`

### 7.4 Primary operator

- `GCNConv`

### 7.5 Sensitivity dimensions

Global hyperparameters:

- learning rate: `0.001, 0.003, 0.005, 0.01`
- dropout: `0.1, 0.3, 0.5, 0.7`
- hidden dimension: `32, 64, 128`
- weight decay: `1e-5, 5e-5, 1e-4, 1e-3`
- depth `L`: `2, 3, 4, 5, 6`
- branch count `B`: `1..8`

Matrix-control hyperparameters:

- `gate_init`: `0.2, 0.5, 0.8, 0.95`
- `topk_ratio`: `0.25, 0.5, 0.75`
- `sparse_lambda`: `0.01, 0.02, 0.05, 0.1`

### 7.6 What to report

- mean/std accuracy
- best epoch
- runtime
- gate value statistics
- residual active ratio
- branch diversity summary

### 7.7 Interpretation target

This section should answer:

- whether the structure is stable under ordinary hyperparameter changes
- whether matrix-gated behavior is highly sensitive to sparsity/gate choices
- whether the strongest setting is also the most stable setting

## 8. Mechanism and representation analysis

### 8.1 Goal

Move beyond final accuracy and explain why some topologies work better on some settings.

### 8.2 Priority targets

Run mechanism analysis first on representative settings, not on the full benchmark matrix.

Recommended first targets:

- `PROTEINS + GCNConv`
- `DD + GCNConv`
- strongest one-axis model on each target
- strongest matrix model on each target

### 8.3 Required questions

The mechanism analysis should support answers to:

- does matrix reuse improve gradient flow
- does matrix reuse diversify branches or collapse them
- does stronger layer continuity correlate with better accuracy
- does branch similarity rise or fall when performance improves
- does sparse/gated control suppress noisy residual traffic

## 9. Intermediate artifacts that must be saved

These are the non-negotiable saved artifacts for later interpretation.

### 9.1 Run-level outputs

Save for every run:

- `result.json`
- `history.json`
- `checkpoint.pt`
- `config.json`

### 9.2 Prediction outputs

Save for paper-facing runs:

- `predictions.csv`
- `logits.npy`
- `labels.npy`
- `preds.npy`
- `graph_embeddings.npz`

Use them for:

- confusion matrices
- calibration or class-separation checks
- embedding visualization
- label-wise error analysis

### 9.3 Layer states

Save for representative targets:

- `layer_states.pt`

Expected contents:

- node states by branch and layer
- graph states by branch and layer
- final branch graph embeddings

Use them for:

- depth continuity analysis
- branch similarity analysis
- branch diversity analysis
- CKA and cosine analysis

### 9.4 Residual-path statistics

Save:

- `residual_stats.json`

Expected contents:

- residual source count
- residual norm before filtering
- residual norm after filtering
- active ratio
- gate value

Use them for:

- dense vs sparse residual interpretation
- residual traffic magnitude analysis
- residual saturation / suppression analysis

### 9.5 Gradient diagnostics

Save:

- `gradient_by_layer.csv`

Expected contents:

- parameter or layer name
- gradient L2 norm

Use them for:

- depth-wise gradient stability
- branch-wise gradient concentration
- whether matrix reuse helps optimization

### 9.6 Similarity analysis

Save:

- `cosine_branch.csv`
- `cosine_depth.csv`
- `cka_branch.csv`
- `cka_depth.csv`

Use them for:

- branch similarity
- adjacent-depth continuity
- whether high continuity corresponds to better performance
- whether matrix reuse increases useful diversity

### 9.7 Branch diversity

Save:

- `branch_diversity.json`

Expected contents:

- mean pairwise branch distance
- max pairwise branch distance
- branch-pair distance table

Use them for:

- branch collapse checks
- branch specialization interpretation
- branch-count ablation explanation

## 10. How to interpret the intermediate data

This section is the bridge from saved files to paper claims.

### 10.1 Hidden states

Use hidden node states and graph states to answer:

- does each branch learn something distinct
- does depth progressively refine or merely repeat representations
- does matrix reuse stabilize later layers

### 10.2 Gradients

Use gradient norms to answer:

- whether deeper models suffer gradient decay
- whether matrix reuse improves deep optimization
- whether some branches dominate the optimization signal

### 10.3 Cosine similarity

Use cosine similarity to answer:

- whether branches collapse to near-identical summaries
- whether adjacent layers become too similar
- whether sparse/gated control changes branch agreement

### 10.4 CKA

Use CKA to answer:

- whether internal representations remain highly continuous across depth
- whether stronger continuity corresponds to better final accuracy
- whether matrix reuse leads to productive representation changes rather than pure preservation

### 10.5 Residual-path statistics

Use residual statistics to answer:

- whether residual injection is dense but weak
- whether sparse/gated modes suppress low-value residual traffic
- whether matrix reuse is effective because of better topology or simply because of more signal volume

### 10.6 Branch diversity

Use branch diversity to answer:

- whether extra branches genuinely diversify representations
- whether performance gains from larger `B` are real or just extra capacity
- whether some datasets favor more diversity than others

## 11. Main-paper vs appendix split

### 11.1 Main paper

Prefer to keep in the main paper:

- main benchmark
- structure ablation
- branch-count ablation
- residual-control ablation
- one focused mechanism section on gradients / similarity / CKA

### 11.2 Appendix

Move to appendix:

- extended parameter sensitivity tables
- additional operators if too many figures accumulate
- extra branch-count curves
- secondary datasets for mechanism analysis
- full raw similarity matrices

## 12. Recommended execution order

1. complete main benchmark on `PROTEINS/DD/ENZYMES + GCNConv`
2. complete all five models for those settings across five folds
3. summarize benchmark and generate main figures
4. run branch-count ablation on `PROTEINS` and `DD`
5. run residual-control ablation on representative targets
6. run parameter sensitivity on `MatrixRes` and `MatrixResGated`
7. export and interpret mechanism artifacts
8. only then expand to more operators and supplementary datasets

## 13. Minimum commands that should exist

Single run:

```bash
conda activate pyg
python scripts/run_single.py --dataset PROTEINS --model MatrixRes --operator GCNConv --fold 0
```

Main benchmark:

```bash
conda activate pyg
python scripts/run_benchmark.py --dataset_group main --operators GCNConv --folds 0 1 2 3 4
```

Branch ablation:

```bash
conda activate pyg
python scripts/run_branch_ablation.py --dataset PROTEINS --operator GCNConv --branches 1 2 3 4 5 6 7 8
```

Benchmark summary:

```bash
conda activate pyg
python scripts/summarize_benchmark.py
```

Figure export:

```bash
conda activate pyg
python scripts/generate_suite_figures.py
```

## 14. Completion criteria

This experiment checklist is only satisfied when:

- every main benchmark run has result, config, checkpoint, and history files
- every representative mechanism target has layer states, gradient, cosine, CKA, and branch-diversity outputs
- benchmark summary CSV is reproducible from log files
- paper-facing figures are reproducible from summary CSV
- main claims can be traced back to saved artifacts rather than informal observations
