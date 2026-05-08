# Results Synthesis for Matrix-ResGNN

Source version: `records/LATEST`. This document organizes the benchmark results, ablations, parameter sensitivity, tuned candidates, and mechanism metrics for manuscript writing.

## 1. Main Benchmark

- Scope: 6 datasets x 4 operators x 5 models x 5 folds = 600 benchmark runs.
- Models: `Plain`, `VerticalRes`, `HorizontalRes`, `MatrixRes`, `MatrixResGated`.
- Operators: `GCNConv`, `GATConv`, `SAGEConv`, `GINConv`.
- Primary metric: five-fold mean best test accuracy with fold-level standard deviation.

### Overall Winner Counts

| Model | Wins / 24 | Mean Rank | Interpretation |
|---|---:|---:|---|
| `Plain` | 5 | 3.04 | Strong on some small/easy settings; useful baseline, not consistently dominated. |
| `VerticalRes` | 2 | 3.38 | Least frequent winner; depth-only reuse helps selectively. |
| `HorizontalRes` | 3 | 3.54 | Competitive when lateral branch exchange is enough, especially selected protein settings. |
| `MatrixRes` | 9 | 2.12 | Best overall by wins and mean rank; strongest general residual topology. |
| `MatrixResGated` | 5 | 2.92 | Second-tier overall; useful when sparse control helps residual traffic. |

Main conclusion: `MatrixRes` is the strongest overall topology, but every model wins at least two dataset-operator settings. This supports a dataset/operator-dependent claim rather than a universal-winner claim.

### Winners by Dataset and Operator

| Dataset | GCNConv | GATConv | SAGEConv | GINConv |
|---|---|---|---|---|
| PROTEINS | `Plain` (0.7044) | `HorizontalRes` (0.7116) | `Plain` (0.6999) | `HorizontalRes` (0.7080) |
| DD | `HorizontalRes` (0.7181) | `MatrixRes` (0.7164) | `MatrixRes` (0.7198) | `VerticalRes` (0.7224) |
| ENZYMES | `MatrixResGated` (0.2933) | `Plain` (0.2717) | `VerticalRes` (0.2883) | `Plain` (0.3100) |
| MUTAG | `MatrixRes` (0.7609) | `MatrixResGated` (0.7555) | `MatrixRes` (0.7504) | `MatrixResGated` (0.8081) |
| AIDS | `MatrixRes` (0.8365) | `MatrixRes` (0.8875) | `Plain` (0.8850) | `MatrixResGated` (0.9280) |
| Mutagenicity | `MatrixRes` (0.7784) | `MatrixRes` (0.7909) | `MatrixResGated` (0.8031) | `MatrixRes` (0.8024) |

### GCNConv Detailed Slice

| Dataset | Plain | VerticalRes | HorizontalRes | MatrixRes | MatrixResGated | Winner |
|---|---:|---:|---:|---:|---:|---|
| PROTEINS | 0.7044 +/- 0.0387 | 0.6993 +/- 0.0340 | 0.6992 +/- 0.0399 | 0.6973 +/- 0.0324 | 0.6886 +/- 0.0397 | `Plain` |
| DD | 0.7053 +/- 0.0517 | 0.7143 +/- 0.0260 | 0.7181 +/- 0.0296 | 0.7127 +/- 0.0365 | 0.7141 +/- 0.0359 | `HorizontalRes` |
| ENZYMES | 0.2683 +/- 0.0343 | 0.2550 +/- 0.0455 | 0.2633 +/- 0.0584 | 0.2750 +/- 0.0577 | 0.2933 +/- 0.0470 | `MatrixResGated` |
| MUTAG | 0.7556 +/- 0.0439 | 0.7451 +/- 0.0695 | 0.7343 +/- 0.0433 | 0.7609 +/- 0.0715 | 0.7556 +/- 0.0721 | `MatrixRes` |
| AIDS | 0.8215 +/- 0.0046 | 0.8210 +/- 0.0233 | 0.8130 +/- 0.0297 | 0.8365 +/- 0.0412 | 0.8150 +/- 0.0231 | `MatrixRes` |
| Mutagenicity | 0.7644 +/- 0.0188 | 0.7667 +/- 0.0212 | 0.7738 +/- 0.0134 | 0.7784 +/- 0.0185 | 0.7701 +/- 0.0165 | `MatrixRes` |

## 2. Branch-Count Ablation

- Scope: `PROTEINS` and `DD`, `GCNConv`, models `HorizontalRes`, `MatrixRes`, `MatrixResGated`, branch count B=1..8, five folds.
- Purpose: test whether multi-branch residual reuse improves monotonically with more branches, or has a useful operating region.

| Dataset | Model | Best B | Best Acc | Worst B | Worst Acc | Pattern |
|---|---|---:|---:|---:|---:|---|
| PROTEINS | `HorizontalRes` | 4 | 0.7125 +/- 0.0286 | 1 | 0.6801 +/- 0.0518 | moderate branch budget |
| PROTEINS | `MatrixRes` | 6 | 0.7062 +/- 0.0210 | 1 | 0.6909 +/- 0.0586 | moderate branch budget |
| PROTEINS | `MatrixResGated` | 3 | 0.7098 +/- 0.0219 | 4 | 0.6711 +/- 0.0471 | moderate branch budget |
| DD | `HorizontalRes` | 2 | 0.7275 +/- 0.0304 | 4 | 0.7071 +/- 0.0302 | small branch budget |
| DD | `MatrixRes` | 2 | 0.7206 +/- 0.0411 | 7 | 0.7071 +/- 0.0390 | small branch budget |
| DD | `MatrixResGated` | 1 | 0.7283 +/- 0.0244 | 7 | 0.7053 +/- 0.0456 | small branch budget |

Branch-count conclusion: performance is non-monotonic. PROTEINS tolerates moderate branch expansion, while DD prefers B=1 or B=2. Extra branches add residual paths and optimization burden, not just capacity.

## 3. Mechanism Metrics: Similarity, Diversity, Gradients, Residual Traffic

Mechanism metrics are mainly used to explain the branch-count ablation. Key meanings:

- `mean_pairwise_distance`: branch diversity. Larger means branches separate more.
- `mean_cosine_branch`: branch cosine similarity. Larger means branches remain redundant.
- `mean_cka_branch`: branch CKA similarity. Larger means representation subspaces are aligned.
- `mean_grad_norm`: average representative gradient norm. Very small values indicate weaker optimization signal.
- `residual_count_total`: total residual inputs across residual sites; grows with branch count/topology.
- `active_ratio_mean`: fraction of nonzero residual entries after filtering; useful for sparse/gated variants.

### Mechanism Snapshot at Each Best Branch Count

| Dataset | Model | Best B | Acc | Diversity | Branch Cosine | Branch CKA | Grad Norm | Residual Count | Active Ratio | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PROTEINS | `HorizontalRes` | 4 | 0.7125 | 1.1450 | 0.8417 | 0.9797 | 0.0157 | 24.0 | 0.2519 | 0.7911 |
| PROTEINS | `MatrixRes` | 6 | 0.7062 | 3.7755 | 0.9173 | 0.9634 | 0.0182 | 88.0 | 0.4273 | 0.7757 |
| PROTEINS | `MatrixResGated` | 3 | 0.7098 | 1.3555 | 0.9790 | 0.9988 | 0.0316 | 37.0 | 0.3121 | 0.7833 |
| DD | `HorizontalRes` | 2 | 0.7275 | 0.1687 | 0.9969 | 1.0000 | 0.0732 | 6.0 | 0.3492 | 0.8375 |
| DD | `MatrixRes` | 2 | 0.7206 | 0.1834 | 0.9991 | 1.0000 | 0.0796 | 14.0 | 0.5384 | 0.8222 |
| DD | `MatrixResGated` | 1 | 0.7283 | 0.0000 | 1.0000 | 1.0000 | 0.0821 | 2.0 | 0.2352 | 0.8221 |

### Selected Mechanism Interpretation

- `PROTEINS + HorizontalRes`: best at B=4 (0.7125). Diversity changes from 0.0000 at B=1 to 1.1450 at best B and 1.5441 at B=8; branch cosine changes from 1.0000 to 0.8417 and 0.7450.
- `PROTEINS + MatrixRes`: best at B=6 (0.7062). Diversity changes from 0.0000 at B=1 to 3.7755 at best B and 3.9679 at B=8; branch cosine changes from 1.0000 to 0.9173 and 0.8792.
- `PROTEINS + MatrixResGated`: best at B=3 (0.7098). Diversity changes from 0.0000 at B=1 to 1.3555 at best B and 2.9356 at B=8; branch cosine changes from 1.0000 to 0.9790 and 0.8138.
- `DD + HorizontalRes`: best at B=2 (0.7275). Diversity changes from 0.0000 at B=1 to 0.1687 at best B and 1.4006 at B=8; branch cosine changes from 1.0000 to 0.9969 and 0.7614.
- `DD + MatrixRes`: best at B=2 (0.7206). Diversity changes from 0.0000 at B=1 to 0.1834 at best B and 2.3278 at B=8; branch cosine changes from 1.0000 to 0.9991 and 0.9129.
- `DD + MatrixResGated`: best at B=1 (0.7283). Diversity changes from 0.0000 at B=1 to 0.0000 at best B and 1.5389 at B=8; branch cosine changes from 1.0000 to 1.0000 and 0.8664.

Mechanism conclusion: useful branch expansion requires separation without excessive redundancy or weak gradients. Diversity alone is insufficient; when diversity continues increasing but accuracy declines, the added branches likely create over-fragmentation or residual traffic rather than useful specialization.

## 4. Parameter Sensitivity

- Scope: fold-0 scans for `MatrixRes` and `MatrixResGated` on PROTEINS and DD.
- Purpose: identify candidate operating regions, not final full-fold claims.

| Dataset | Model | Baseline | Best Setting | Best Acc | Notes |
|---|---|---:|---|---:|---|
| DD | `MatrixRes` | 0.7468 | `dim=32` | 0.7595 | candidate setting |
| DD | `MatrixResGated` | 0.7300 | `dim=32` | 0.7511 | candidate setting |
| PROTEINS | `MatrixRes` | 0.6637 | `baseline` | 0.6637 | default remains best in fold-0 scan |
| PROTEINS | `MatrixResGated` | 0.6726 | `sparse_lambda=0.02` | 0.6951 | candidate setting |

Sensitivity conclusion: PROTEINS shows a fold-0 benefit from mild sparsity for MatrixResGated (`sparse_lambda=0.02`), while DD prefers lighter/conservative settings such as `dim=32`, `gate_init=0.2`, or `lr=0.001`. These are hypothesis-generating results.

## 5. Five-Fold Tuned Candidate Checks

| Candidate | Dataset | Acc | Mean Loss | Params | Interpretation |
|---|---|---:|---:|---:|---|
| `DD_dim_32` | DD | 0.7215 +/- 0.0330 | 0.5717 | 15,043 | parameter-efficient but not clearly better than default |
| `DD_gate_init_0.2` | DD | 0.7131 +/- 0.0213 | 0.5820 | 42,371 | lower gate is less stable over five folds |
| `DD_lr_0.001` | DD | 0.7181 +/- 0.0336 | 0.5737 | 42,371 | competitive with default; not a decisive full-fold gain |
| `PROTEINS_sparse_lambda_0.02` | PROTEINS | 0.6909 +/- 0.0447 | 0.6104 | 38,339 | fold-0 gain weakens under full validation |

Tuned-candidate conclusion: five-fold reruns prevent over-claiming from fold-0 sensitivity. The strongest practical candidate is `DD_dim_32`, which lowers parameters substantially while staying competitive.

## 6. Recommended Paper Placement

- Main Results: include overall winner counts, GCNConv detailed table, model win-count figure.
- Ablation Results: include branch-count table/curve and explain non-monotonic behavior.
- Mechanism Results: keep diversity, branch cosine, gradient norm, and residual active ratio in main text; move full CKA/depth cosine matrices to Appendix.
- Appendix: include full benchmark summary, fold rows, parameter scans, tuned candidates, CKA, residual stats, and queue/completeness records.

