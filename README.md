# Matrix-ResGNN-GraphClassification

This repository is the clean restart for a graph-classification paper centered on structured residual connectivity.

The paper will study a unified residual family on graph classification benchmarks:

- `Plain`
- `Vertical-Res`
- `Horizontal-Res`
- `Matrix-Res`
- `Matrix-Res (sparse/gated)`

The core task remains graph classification. The initial benchmark package will reuse the current graph-classification setup from `cross_residual_gnn`, but the manuscript, naming, and method framing will be rebuilt around a two-axis residual-connectivity view rather than the previous cross-residual wording.

## Repository layout

- `docs/`: planning, writing notes, and task tracking
- `paper/`: new manuscript sources
- `src/`: model and training code
- `configs/`: experiment configuration files
- `scripts/`: reusable entry scripts
- `figures/`: generated figures for the paper
- `records/`: outputs, summaries, and curated experiment results

## Working assumptions

- Use the Conda environment `pyg` for Python execution.
- Reuse code only when it supports the new matrix-residual framing cleanly.
- Do not copy old paper text directly; the new paper should be written as an independent manuscript.

## Immediate next step

See [docs/WORK_PLAN.md](/ds1/workspace/ai/Matrix-ResGNN-GraphClassification/docs/WORK_PLAN.md:1).
