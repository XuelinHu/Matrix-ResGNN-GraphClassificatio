# Submission Figure List

英文正式稿中的图片已按首次出现顺序编号。投稿图片统一由对应 PDF 源文件以 300 DPI 转换为 PNG 格式。

| No. | PNG file | English name | 中文名称 | Source |
| --- | --- | --- | --- | --- |
| Figure 1 | `Figure_01_Architecture_Overview.png` | Architecture Overview | 残差邻域架构总览 | `figures/cr_gnn_graph_architecture.pdf` |
| Figure 2 | `Figure_02_Model_Level_Winner_Counts.png` | Model-Level Winner Counts | 模型级胜出次数 | `figures/exp/fig_model_win_counts.pdf` |
| Figure 3 | `Figure_03_GCNConv_Benchmark.png` | GCNConv Benchmark | GCNConv 基准切片 | `figures/exp/fig_main_benchmark_gcnconv.pdf` |
| Figure 4 | `Figure_04_Synthetic_Multiclass_Scaling.png` | Synthetic Multiclass Scaling | 合成多分类扩展趋势 | `figures/exp/fig_synthetic_multiclass_scaling.pdf` |
| Figure 5 | `Figure_05_Branch_Count_Ablation.png` | Branch-Count Ablation | 分支数消融 | `figures/exp/fig_branch_count_ablation.pdf` |
| Figure 6 | `Figure_06_Mechanism_Summary.png` | Mechanism Summary | 机制分析摘要 | `figures/exp/fig_mechanism_branch_dynamics.pdf` |
| Figure 7 | `Figure_07_MatrixResGated_Sensitivity.png` | MatrixResGated Sensitivity | MatrixResGated 灵敏度扫描 | `figures/exp/fig_matrixresgated_sensitivity.pdf` |

## Captions

**Figure 1. Architecture Overview.** Architecture overview of the compared residual neighborhoods on the branch-by-layer grid. The same message-passing backbone and graph-level readout are kept fixed while the residual routes are changed.

**Figure 2. Model-Level Winner Counts.** Model-level winner counts across the 24 dataset--operator combinations.

**Figure 3. GCNConv Benchmark.** GCNConv benchmark at \(B=3\). Bars show five-fold mean best test accuracy with fold-level standard-deviation error bars.

**Figure 4. Synthetic Multiclass Scaling.** Synthetic multi-class scaling from two to eight classes. Each point averages five synthetic graph families and four message-passing operators.

**Figure 5. Branch-Count Ablation.** Branch-count ablation for HorizontalRes, MatrixRes, and MatrixResGated on PROTEINS and DD under GCNConv. The curves are not monotonic: accuracy improves only within an effective operating region before saturating or declining.

**Figure 6. Mechanism Summary.** Mechanism summary linking branch-count accuracy to branch diversity, branch cosine similarity, branch CKA, and mean gradient norm.

**Figure 7. MatrixResGated Sensitivity.** Fold-0 MatrixResGated sensitivity scan at \(B=3\). The scan identifies candidate operating regions but is not treated as final hyperparameter evidence.
