# TODO Handoff

## Current status

- manuscript skeleton completed:
  - `paper/MANUSCRIPT_DRAFT.md`
  - complete Markdown paper draft
  - abstract / introduction / related work / methods / experimental setup / results / discussion / limitations / conclusion / appendices
  - in-text tables for benchmark, branch ablation, and sensitivity results
- bibliography draft completed:
  - `paper/references.bib`
  - citation keys in `paper/MANUSCRIPT_DRAFT.md` verified against BibTeX entries
- main-text figure captions completed:
  - `paper/FIGURE_CAPTIONS.md`
  - Figure 1-4 captions aligned with current figure plan
- main benchmark completed:
  - `PROTEINS / DD / ENZYMES`
  - `Plain / VerticalRes / HorizontalRes / MatrixRes / MatrixResGated`
  - `GCNConv`
  - `5 folds`
- branch ablation completed:
  - `PROTEINS / DD`
  - `HorizontalRes / MatrixRes / MatrixResGated`
  - `B=1..8`
- first-batch parameter sensitivity completed:
  - `PROTEINS / DD`
  - `MatrixRes / MatrixResGated`
  - `fold0`
  - `B=3`
- mechanism summaries and compact tables completed
- paper-facing draft notes and figures completed

## Key files

### Core summaries

- `records/LATEST/summaries/benchmark_summary.csv`
- `records/LATEST/summaries/branch_ablation_summary.csv`
- `records/LATEST/summaries/parameter_sensitivity_summary.csv`
- `records/LATEST/summaries/mechanism_compact_summary.csv`

### Writing notes

- `paper/MANUSCRIPT_DRAFT.md`
- `paper/FIGURE_CAPTIONS.md`
- `paper/references.bib`
- `paper/METHOD_NOTE.md`
- `paper/RESULTS_DISCUSSION_DRAFT.md`
- `paper/FIGURE_PLAN.md`
- `docs/RESULT_INTERPRETATION.md`

### Plot scripts

- `scripts/generate_suite_figures.py`
- `scripts/generate_ablation_figures.py`
- `scripts/generate_mechanism_figure.py`

## High-priority next work

1. Convert the Markdown manuscript into the target submission format when venue is known:
   - LaTeX template
   - bibliography file
   - figure includes
   - table environment for benchmark and ablation summaries

2. Decide the main-text benchmark scope:
   - likely keep `PROTEINS + DD + ENZYMES`
   - keep supplementary datasets for appendix unless later expanded

3. Decide whether to run a second parameter round:
   - `MatrixResGated` on `PROTEINS` with `sparse_lambda` near `0.02`
   - `MatrixResGated` on `DD` with `lr=0.001`, `dim=32`, `gate_init=0.2`

4. If expanding experiments, run non-GCNConv operator checks before broad dataset expansion:
   - `GINConv` or `GraphSAGE`
   - same folds and branch settings for direct comparability

5. Before submission, strengthen the bibliography and related work:
   - add venue-specific citation style
   - add oversquashing references if the final introduction keeps that framing
   - add any directly comparable residual/multi-branch GNN baselines required by reviewers

## Main paper messages

### Branch-count message

- branch count helps first because it creates useful branch diversity
- too many branches hurt because:
  - residual traffic grows quickly
  - branch similarity remains partly high
  - optimization weakens
- `PROTEINS` prefers a moderate-to-larger branch budget
- `DD` prefers a smaller branch budget

### MatrixResGated tuning message

- `PROTEINS`:
  - mild sparsification helps
  - `sparse_lambda=0.02` is the strongest first-batch setting
- `DD`:
  - lighter and more conservative settings work better
  - `lr=0.001`
  - `dim=32`
  - `gate_init=0.2`

### Main mechanism indicators for the paper

- keep in main text:
  - `branch diversity`
  - `branch cosine similarity`
  - `mean gradient norm`
- move to appendix or secondary support:
  - `CKA`
  - `active_ratio_mean`
  - `residual_count_total`
  - `residual_norm_pre_total`
  - `residual_norm_post_total`

## Compressed figure plan

### Figure 1

- file: `figures/exp/fig_main_benchmark_gcnconv.pdf`
- purpose:
  - main benchmark ranking across core datasets

### Figure 2

- file: `figures/exp/fig_branch_count_ablation.pdf`
- purpose:
  - show rise-then-fall behavior of `B`
  - compare `HorizontalRes / MatrixRes / MatrixResGated`
  - emphasize `PROTEINS` vs `DD` difference

### Figure 3

- file: `figures/exp/fig_matrixresgated_sensitivity.pdf`
- purpose:
  - show usable parameter region of `MatrixResGated`
  - emphasize dataset-specific tuning

### Figure 4

- file: `figures/exp/fig_mechanism_branch_dynamics.pdf`
- purpose:
  - explain branch-count trend with:
    - accuracy
    - branch diversity
    - branch cosine
    - mean gradient norm

## Useful commands

```bash
conda run -n pyg python scripts/summarize_benchmark.py
conda run -n pyg python scripts/summarize_branch_ablation.py
conda run -n pyg python scripts/summarize_parameter_sensitivity.py
conda run -n pyg python scripts/summarize_mechanism_artifacts.py
conda run -n pyg python scripts/summarize_mechanism_compact.py
conda run -n pyg python scripts/generate_suite_figures.py
conda run -n pyg python scripts/generate_ablation_figures.py
conda run -n pyg python scripts/generate_mechanism_figure.py
```

## If restarting from a fresh session

1. Open this file first.
2. Then read:
   - `paper/MANUSCRIPT_DRAFT.md`
   - `paper/FIGURE_CAPTIONS.md`
   - `paper/references.bib`
   - `paper/METHOD_NOTE.md`
   - `paper/RESULTS_DISCUSSION_DRAFT.md`
   - `paper/FIGURE_PLAN.md`
3. Use the summary CSV files as the single source of truth.
4. Continue with manuscript writing before expanding experiments again.
