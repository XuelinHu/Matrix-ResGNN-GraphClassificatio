# Supporting Evidence CSV Package

The following Excel-compatible CSV tables are prepared for upload as supporting evidence. The files are numbered in submission order. Real-data results and synthetic multi-class results use distinct file-name prefixes.

| No. | File | Scope | Contents |
|---:|---|---|---|
| 01 | `01_real_data_benchmark_summary.csv` | Real-data benchmark | 120 five-fold model-level aggregates across six datasets, four message-passing operators, and five model families. Includes accuracy, loss, best epoch, runtime, parameter count, branch count, and residual mode. |
| 02 | `02_real_data_branch_ablation_summary.csv` | Real-data branch-count ablation | 48 five-fold aggregates for HorizontalRes, MatrixRes, and MatrixResGated on PROTEINS and DD with branch counts from one to eight. |
| 03 | `03_real_data_parameter_sensitivity_summary.csv` | Real-data sensitivity scan | 45 fold-0 scan rows for MatrixRes and MatrixResGated on PROTEINS and DD, covering hidden dimension, dropout, learning rate, sparse strength, and gate initialization. |
| 04 | `04_real_data_tuned_candidate_summary.csv` | Real-data five-fold candidate reruns | Four MatrixResGated candidate checks under GCNConv: three on DD and one on PROTEINS. |
| 05 | `05_real_data_mechanism_compact_summary.csv` | Real-data mechanism analysis | 57 diagnostic rows from PROTEINS, DD, and ENZYMES. Includes branch diversity, residual statistics, active ratios, gate values, gradients, cosine similarities, and CKA values. |
| 06 | `06_synthetic_multiclass_benchmark_summary.csv` | Synthetic multi-class benchmark | 700 five-fold model-level aggregates spanning five graph families, two to eight structural classes, four message-passing operators, and five model families. Includes accuracy, macro-F1, chance-normalized accuracy, loss, runtime, parameter count, branch count, and residual mode. |

These files correspond to the supporting-evidence list in Appendix A.1 of the manuscript.
 