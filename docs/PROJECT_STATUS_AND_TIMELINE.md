# MatrixResGNNGraphClassification Project Status and Timeline

Updated: `2026-06-01`

This document is the consolidated reading entry point for the repository. It summarizes the current manuscript state, experiment milestones, submission package, repository identity, and reproducibility entry points.

## 0. Repository identity

The paper submission should refer to the local repository identity below:

- Repository name: `MatrixResGNNGraphClassification`
- Canonical local branch: `master`
- Remote: `git@github.com:XuelinHu/MatrixResGNNGraphClassification.git`

## 1. Current paper scope

The paper studies residual connectivity for graph classification under a unified `branch x layer` view. The compared model families are:

- `Plain`
- `VerticalRes`
- `HorizontalRes`
- `MatrixRes`
- `MatrixResGated`

The manuscript is organized around three questions:

1. Does residual topology affect graph-classification performance under a unified protocol?
2. How do branch count, residual traffic, and mechanism metrics explain performance changes?
3. When is matrix-style residual reuse effective on real data and controlled synthetic multi-class structures?

## 2. Frozen experiment packages

### Real-data benchmark

- Version: `records/LATEST`
- Scope: 6 datasets x 4 operators x 5 model families x 5 folds = 600 completed runs
- Datasets: `PROTEINS`, `DD`, `ENZYMES`, `MUTAG`, `AIDS`, `Mutagenicity`
- Operators: `GCNConv`, `GATConv`, `SAGEConv`, `GINConv`

### Real-data follow-up studies

- Branch-count ablation: 48 five-fold aggregates on `PROTEINS` and `DD`, with `B=1..8`
- Parameter sensitivity: 45 fold-0 scan rows
- Tuned-candidate reruns: 4 five-fold `MatrixResGated` checks
- Mechanism analysis: 57 compact diagnostic rows

### Synthetic multi-class benchmark

- Version: `records/SYN_MULTI_FULL`
- Scope: 5 graph families x 7 class counts x 4 operators x 5 model families x 5 folds = 3,500 completed runs
- Graph families: `ER`, `BA`, `SBM`, `WS`, `REGULAR`
- Class counts: `C=2..8`

## 3. Submission artifacts

### Manuscripts

- English source: `paper/main.tex`
- English PDF: `paper/main.pdf`
- Chinese confirmation source: `paper/main_zh.tex`
- Chinese confirmation PDF: `paper/main_zh.pdf`

### Figures

- Editable architecture source: `figures/cr_gnn_graph_architecture.drawio`
- Architecture exports: `figures/cr_gnn_graph_architecture.svg`, `figures/cr_gnn_graph_architecture.pdf`
- Numbered PNG package: `paper/submission_figures_png/`
- PNG package archive: `paper/submission_figures_png.zip`

### Supporting-evidence CSV package

- Upload folder: `paper/supporting_evidence_csv/`
- Package manifest: `paper/supporting_evidence_csv/README.md`
- Numbered CSV files: `01` to `06`
- Naming rule: real-data files use `real_data`; generated-data files use `synthetic_multiclass`

## 4. Key milestones

| Date | Milestone |
|---|---|
| `2026-05-06` | Started the first real-data benchmark stage on `PROTEINS`, `DD`, and `ENZYMES` with `GCNConv`. |
| `2026-05-07` | Started branch-count ablation and fold-0 parameter sensitivity scans. |
| `2026-05-08` | Added the PeerJ manuscript template, result summaries, figures, citations, and Chinese confirmation PDF. |
| `2026-05-11` | Revised the manuscript after external review and iterated the architecture figure. |
| `2026-05-31` | Added the controlled synthetic multi-class benchmark results and prepared the numbered PNG submission package. |
| `2026-06-01` | Regenerated the architecture exports, improved PDF typography, organized the numbered supporting-evidence CSV package, and consolidated documentation. |

## 5. Reproducibility entry points

Use the `pyg` Conda environment for Python scripts:

```powershell
conda run -n pyg python scripts/summarize_benchmark.py
conda run -n pyg python scripts/summarize_branch_ablation.py
conda run -n pyg python scripts/summarize_parameter_sensitivity.py
conda run -n pyg python scripts/summarize_mechanism_compact.py
conda run -n pyg python scripts/generate_suite_figures.py
conda run -n pyg python scripts/generate_ablation_figures.py
conda run -n pyg python scripts/generate_mechanism_figure.py
```

Compile the manuscripts from `paper/`:

```powershell
xelatex -interaction=nonstopmode main.tex
bibtex main
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex

xelatex -interaction=nonstopmode main_zh.tex
bibtex main_zh
xelatex -interaction=nonstopmode main_zh.tex
xelatex -interaction=nonstopmode main_zh.tex
```

## 6. Script entry points

The script inventory is maintained in `scripts/README.md`. The main groups are:

- Experiment runners: `run_*.py` and `run_*.sh`
- Summary generators: `summarize_*.py`
- Figure generators: `generate_*.py`
- Export and check utilities: `export_*.py`, `check_*.py`

The old planning, handoff, review, and draft-analysis documents have been removed from `docs/` so that this directory contains only current submission-facing documentation.
 