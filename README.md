# Matrix-ResGNN-GraphClassification

This repository contains the implementation, frozen experiment records, manuscript sources, and submission artifacts for a graph-classification paper centered on structured residual connectivity.

The paper studies a unified residual family on graph-classification benchmarks:

- `Plain`
- `VerticalRes`
- `HorizontalRes`
- `MatrixRes`
- `MatrixResGated`

The core task is graph classification. The manuscript frames residual routing as a two-axis `branch x layer` design problem and evaluates real-data and controlled synthetic multi-class benchmarks.

## Repository layout

- `docs/`: consolidated project status, historical planning notes, and task records
- `paper/`: manuscript sources and submission packages
- `src/`: model and training code
- `configs/`: experiment configuration files
- `scripts/`: reusable entry scripts
- `figures/`: generated figures for the paper
- `records/`: outputs, summaries, and curated experiment results

## Current artifacts

- English manuscript: `paper/main.tex`, `paper/main.pdf`
- Chinese confirmation manuscript: `paper/main_zh.tex`, `paper/main_zh.pdf`
- Numbered PNG figure package: `paper/submission_figures_png/`
- Numbered supporting-evidence CSV package: `paper/supporting_evidence_csv/`

## Environment

Use the Conda environment `pyg` for Python execution.

## Documentation

See [docs/README.md](docs/README.md) for the current documentation entry point and the consolidated project timeline.

## PeerJ references
- https://peerj.com/articles/cs-3773/
- https://peerj.com/articles/cs-3762/
- https://www.overleaf.com/latex/templates/latex-template-for-peerj-journal-and-pre-print-submissions/ptdwfrqxqzbn
- https://peerj.com/about/policies-and-procedures/#discipline-standards
- https://peerj.com/about/author-instructions/#reference-format
