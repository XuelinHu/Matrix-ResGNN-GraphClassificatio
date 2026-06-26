# MatrixResGNNGraphClassification

<p align="center">
  <img height="20" src="https://img.shields.io/badge/python-used-3776AB?logo=python&logoColor=white" />
  <img height="20" src="https://img.shields.io/badge/pytorch-used-EE4C2C?logo=pytorch&logoColor=white" />
  <img height="20" src="https://img.shields.io/badge/pytorch_geometric-used-3C2179" />
  <img height="20" src="https://img.shields.io/badge/numpy-used-013243?logo=numpy&logoColor=white" />
  <img height="20" src="https://img.shields.io/badge/pandas-used-150458?logo=pandas&logoColor=white" />
  <img height="20" src="https://img.shields.io/badge/matplotlib-used-11557C" />
  <img height="20" src="https://img.shields.io/badge/latex-paper-008080?logo=latex&logoColor=white" />
</p>

This repository contains the implementation, frozen experiment records, manuscript sources, and submission artifacts for a graph-classification paper centered on structured residual connectivity.

Repository identity for the paper submission:

- Repository name: `MatrixResGNNGraphClassification`
- Canonical local branch: `master`
- Remote: `git@github.com:XuelinHu/MatrixResGNNGraphClassification.git`

The paper studies a unified residual family on graph-classification benchmarks:

- `Plain`
- `VerticalRes`
- `HorizontalRes`
- `MatrixRes`
- `MatrixResGated`

The core task is graph classification. The manuscript frames residual routing as a two-axis `branch x layer` design problem and evaluates real-data and controlled synthetic multi-class benchmarks.

## Repository layout

- `docs/`: current project status and reproducibility notes
- `paper/`: manuscript sources and submission packages
- `src/`: model and training code
- `configs/`: experiment configuration files
- `scripts/`: runnable experiment, summary, figure, export, and check scripts
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

See [docs/README.md](docs/README.md) for the current documentation entry point. See [scripts/README.md](scripts/README.md) for the script inventory.

## PeerJ references
- https://peerj.com/articles/cs-3773/
- https://peerj.com/articles/cs-3762/
- https://www.overleaf.com/latex/templates/latex-template-for-peerj-journal-and-pre-print-submissions/ptdwfrqxqzbn
- https://peerj.com/about/policies-and-procedures/#discipline-standards
- https://peerj.com/about/author-instructions/#reference-format
