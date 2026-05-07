# Matrix-ResGNN Work Plan

## 1. Project objective

Build a new graph-classification paper around a unified residual-connectivity framework with three structured regimes:

- `Vertical-Res`: same-branch, cross-layer reuse
- `Horizontal-Res`: cross-branch, same-layer exchange
- `Matrix-Res`: local two-dimensional reuse across neighboring branches and neighboring layers

The main contribution is not another `cross-residual` variant. The contribution is a structured view of residual connectivity on a `branch x layer` grid.

## 2. Target model family

- `Plain`
- `Vertical-Res`
- `Horizontal-Res`
- `Matrix-Res`
- `Matrix-Res (sparse/gated)`

Initial implementation assumption:

- number of branches `B = 3`
- later branch-count ablation: `B = 1...8`

## 3. Unified method skeleton

Let the hidden state at branch `b` and layer `l` be `H_b^(l)`.

Base propagation:

```text
Z_b^(l) = phi_b^(l)(H_b^(l-1), A)
```

Unified residual update:

```text
H_b^(l) = Z_b^(l) + sum_{(b', l') in N(b, l)} R_{(b', l')->(b, l)}(H_{b'}^(l'))
```

Where:

- `N(b, l)` is the allowed residual neighborhood
- `R` is the residual transform, initially identity, gated, or sparse

Neighborhood definitions:

- `Vertical-Res`: `N(b, l) = {(b, l-1)}`
- `Horizontal-Res`: `N(b, l) = {(b-1, l), (b+1, l)}`
- `Matrix-Res`: `N(b, l) = {(b, l-1), (b-1, l), (b+1, l), (b-1, l-1), (b+1, l-1)}`

Boundary branches use only valid neighbors.

## 4. Why this project is separate

- The old graph-classification paper will not be submitted.
- The new manuscript may reuse datasets, training protocol, and parts of the codebase.
- The framing changes from `cross-residual comparison` to `structured matrix-style residual connectivity`.
- The method section and paper narrative must be rewritten from scratch for this framing.

## 5. Reuse policy from `cross_residual_gnn`

Allowed to reuse after review:

- dataset loading utilities
- training loop infrastructure
- evaluation utilities
- plotting utilities that are still relevant
- selected architectural code patterns

Must be rebuilt or renamed:

- manuscript text
- model-family naming
- method figures
- result tables
- claims and research questions

## 6. Proposed directory responsibilities

- `src/models/`
  - `plain.py`
  - `vertical_res.py`
  - `horizontal_res.py`
  - `matrix_res.py`
- `src/training/`
  - training, evaluation, checkpoint logic
- `configs/`
  - benchmark configs
  - branch-count ablation configs
  - sparse/gated ablation configs
- `scripts/`
  - dataset runs
  - table aggregation
  - figure generation
- `paper/`
  - manuscript sources
  - appendix sources
- `records/`
  - frozen CSV summaries
  - paper-facing result packages

## 7. Execution order

### Phase A. Research framing

1. Freeze research questions.
2. Finalize the unified formulation and residual neighborhoods.
3. Decide the final model names used in the paper.

### Phase B. Code migration and scaffolding

1. Audit `cross_residual_gnn` for reusable code.
2. Copy only infrastructure that survives the new framing.
3. Build a clean `src/` layout for the five target models.

### Phase C. First benchmark round

1. Reproduce the current graph-classification training pipeline.
2. Run `Plain`, `Vertical-Res`, `Horizontal-Res`, and `Matrix-Res`.
3. Check parameter count, runtime, memory, and result sanity.

### Phase D. Mechanism round

1. Add `Matrix-Res (sparse/gated)`.
2. Run branch-count ablation `B = 1...8`.
3. Test whether sparse/gated control improves stability or efficiency.

### Phase E. Paper package

1. Write new paper title, abstract, and introduction.
2. Build new method figures around the `branch x layer` grid.
3. Generate paper-facing tables and plots.
4. Compile English and Chinese PDFs if both are needed.

## 8. Immediate to-do list

- Create `src/` subdirectories and module placeholders.
- Audit the old repository for code to migrate first.
- Write a short method note describing the `branch x layer` grid view.
- Decide whether the first manuscript will stay on the same dataset package or prune it further.

## 9. Submission rule for this repo

Before claiming paper-ready status, the repo should contain:

- reproducible training entrypoints
- fixed configs for the paper runs
- frozen result summaries
- manuscript sources
- figure-generation scripts
- a top-level submission checklist
