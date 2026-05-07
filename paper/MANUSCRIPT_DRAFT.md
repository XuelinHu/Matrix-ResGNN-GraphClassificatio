# Matrix-Residual Graph Neural Networks for Graph Classification

## Abstract

Residual connectivity is a standard tool for stabilizing deep graph neural networks, but residual design is usually treated as a one-dimensional depth-wise choice. This paper studies residual reuse as a structured design problem over a two-dimensional branch-by-layer grid. Under a shared graph-classification protocol, we compare plain message passing, vertical residual reuse, horizontal branch-wise reuse, local matrix residual reuse, and a sparse/gated matrix variant. Across PROTEINS, DD, and ENZYMES with GCNConv backbones and five-fold evaluation, no single residual topology dominates all datasets: horizontal reuse is strongest on PROTEINS at the default branch count, vertical reuse is strongest on DD, and sparse/gated matrix reuse is strongest on ENZYMES. Branch-count ablations further show a consistent rise-then-fall pattern. Additional branches help when they create useful branch diversity, but too many branches increase residual traffic, preserve redundant branch similarity, and weaken optimization. First-batch sensitivity scans indicate dataset-specific operating regions for controlled matrix reuse: PROTEINS benefits from mild sparsification, while DD favors smaller and more conservative settings. A follow-up five-fold check of selected MatrixResGated settings confirms that the smaller DD configuration improves over the default gated matrix model, while the PROTEINS sparsity candidate remains more variable. These results suggest that residual topology, branch budget, and residual control should be treated jointly rather than as independent architectural choices.

## 1. Introduction

Graph neural networks (GNNs) learn graph representations through iterative neighborhood aggregation and have become a standard model class for node-level and graph-level learning [@kipf2017gcn; @xu2019gin]. For graph classification, the model must build a graph-level representation that preserves useful structural and attribute information across message-passing layers. Increasing depth or adding parallel feature paths can improve expressive capacity, but it also introduces known failure modes such as oversmoothing, redundant representations, unstable optimization, and excessive computation [@li2018deeper; @zhao2020pairnorm].

Residual connections are a common response to these issues. They improve gradient flow and allow later layers to reuse earlier representations, and residual or dense connections have also been adapted to deeper GNNs [@li2019deepgcns]. However, residual connectivity is often treated as a default architectural ingredient rather than as a design object. Most residual GNN variants emphasize depth-wise reuse: a layer receives information from a previous layer on the same stream. This is useful, but it does not exhaust the design space once a model contains multiple branches.

This work studies residual connectivity in multi-branch GNNs from a branch-by-layer perspective. Instead of asking only whether residual connections should be present, we ask where residual signals should flow. A branch-by-layer grid exposes several distinct reuse patterns: vertical reuse along depth, horizontal exchange across branches at the same depth, and local matrix-style reuse that combines depth-wise, branch-wise, and diagonal neighborhoods. This view makes it possible to compare residual topologies while keeping the propagation operator, classifier, and training protocol fixed.

The central hypothesis is that residual topology controls a tradeoff between useful representational diversity and excessive residual traffic. A small branch budget may not provide enough distinct features to exploit cross-branch reuse. A large branch budget may create more diversity, but the additional residual paths can remain redundant or weaken optimization. Sparse or gated residual transforms may help by filtering matrix-style reuse, but their preferred configuration is expected to depend on the dataset.

This paper makes three contributions:

- We formulate vertical, horizontal, and matrix residual GNNs under a unified branch-by-layer residual neighborhood.
- We evaluate these residual families under a fixed graph-classification protocol on PROTEINS, DD, and ENZYMES from TUDataset [@morris2020tudataset], including branch-count ablations on PROTEINS and DD.
- We connect the observed accuracy trends to branch diversity, branch cosine similarity, and gradient norms, showing why branch expansion first helps and then saturates or declines.

## 2. Related Work

### 2.1 Graph Neural Networks for Graph Classification

Message-passing GNNs update node features by aggregating information from graph neighborhoods. GCN introduced an efficient graph convolutional formulation for semi-supervised learning on graphs [@kipf2017gcn], while GIN analyzed the expressive power of neighborhood aggregation and established strong graph-classification baselines [@xu2019gin]. Graph classification commonly evaluates models on benchmark graph datasets such as PROTEINS, DD, and ENZYMES, collected and standardized through TUDataset [@morris2020tudataset].

This paper does not propose a new message-passing operator. Instead, it fixes the base operator to GCNConv in the main benchmark and isolates the effect of residual topology. This controlled setup is intentionally narrower than a full model zoo comparison, but it makes the residual-connectivity question easier to interpret.

### 2.2 Depth, Oversmoothing, and Residual Reuse

Stacking many GNN layers can degrade performance. One explanation is oversmoothing: repeated graph propagation makes node representations increasingly similar [@li2018deeper; @zhao2020pairnorm]. Residual and dense connections are widely used to stabilize deep neural networks and have been adapted to GNNs, including DeepGCNs, which transfer residual and dense connectivity ideas from CNNs to graph convolutional networks [@li2019deepgcns]. Jumping Knowledge Networks aggregate representations from different depths to adaptively reuse layer-wise information [@xu2018jknet].

These approaches show that reuse across depth is valuable, but they primarily focus on depth-wise or layer-wise representation reuse. Our branch-by-layer formulation makes branch-wise and diagonal residual neighborhoods explicit, allowing us to compare one-axis and two-axis reuse patterns under the same training protocol.

### 2.3 Multi-branch Representations

Multi-branch architectures can increase representational diversity by letting different streams learn complementary transformations. In GNNs, this idea appears in multi-hop, multi-scale, ensemble-style, or attention-based variants, although the exact implementation varies by model family. The key issue is that more branches do not automatically produce more useful information. Branches can remain highly correlated, and the extra paths increase parameters, residual traffic, and optimization complexity.

This paper focuses on that tradeoff. It asks whether cross-branch residual reuse creates useful diversity or simply adds redundant residual traffic. The mechanism analysis therefore emphasizes branch diversity, branch cosine similarity, and gradient norms.

## 3. Method

### 3.1 Branch-by-layer notation

Let \(H_b^{(l)}\) denote the hidden state at branch \(b\) and layer \(l\), where \(b \in \{1,\ldots,B\}\) indexes branches and \(l \in \{1,\ldots,L\}\) indexes message-passing depth. Let \(\phi_b^{(l)}(\cdot, A)\) be the graph propagation operator for branch \(b\) at layer \(l\) under graph structure \(A\). The branch-local update is

\[
Z_b^{(l)} = \phi_b^{(l)}(H_b^{(l-1)}, A).
\]

Residual-enhanced propagation is written as

\[
H_b^{(l)} =
Z_b^{(l)} +
\sum_{(b', l') \in \mathcal{N}(b,l)}
R_{(b',l') \rightarrow (b,l)}(H_{b'}^{(l')}).
\]

Here, \(\mathcal{N}(b,l)\) is the allowed residual neighborhood and \(R\) is the residual transform applied before injection. This notation keeps the graph operator and classifier fixed while changing only the residual neighborhood and residual transform.

### 3.2 Residual neighborhoods

The plain model has no residual neighborhood:

\[
\mathcal{N}(b,l) = \emptyset.
\]

VerticalRes uses the previous layer on the same branch:

\[
\mathcal{N}(b,l) = \{(b,l-1)\}.
\]

HorizontalRes exchanges information across adjacent branches at the same depth:

\[
\mathcal{N}(b,l) = \{(b-1,l),(b+1,l)\},
\]

with boundary branches using only valid neighbors.

MatrixRes combines local vertical, horizontal, and diagonal reuse:

\[
\mathcal{N}(b,l) =
\{(b,l-1),(b-1,l),(b+1,l),(b-1,l-1),(b+1,l-1)\}.
\]

This is a local two-dimensional stencil over the branch-by-layer grid, not dense all-to-all residual mixing. The locality constraint is important: it keeps the residual design interpretable and prevents matrix residuals from becoming unconstrained feature mixing.

### 3.3 Sparse and gated matrix reuse

MatrixResGated uses the same matrix neighborhood but controls residual injection through a filtered transform. In the current implementation, residual traffic is controlled through gated or sparse transformations such as

\[
R(U) = \alpha U
\]

or

\[
R(U) = \operatorname{softshrink}_{\lambda}(U).
\]

The purpose of this variant is to test whether matrix-style reuse should be passed densely or selectively. Dense matrix reuse maximizes information flow, but it can also pass redundant or noisy residual signals. Gating and sparsification reduce the effective residual traffic and can therefore act as a control mechanism when the branch-by-layer grid becomes too connected.

### 3.4 Graph-level classifier

All model families use the same graph-classification wrapper. Each branch produces layer-wise node representations through the selected message-passing and residual-update scheme. The final graph representation is obtained through the shared pooling and classifier pipeline used by the repository benchmark. This keeps the comparison focused on residual topology rather than readout design.

## 4. Experimental Setup

### 4.1 Datasets and backbone

We evaluate graph classification on PROTEINS, DD, and ENZYMES from TUDataset [@morris2020tudataset]. All main benchmark runs use GCNConv as the message-passing operator [@kipf2017gcn]. The compared model families are Plain, VerticalRes, HorizontalRes, MatrixRes, and MatrixResGated. Unless otherwise specified, the default branch count is \(B=3\).

### 4.2 Evaluation protocol

The main benchmark uses five folds for each dataset and model. We report mean best test accuracy and standard deviation across folds. We also record test loss, best epoch, runtime, parameter count, and residual diagnostics. The main benchmark uses the same branch count, hidden configuration, optimizer settings, and fold protocol across model families.

The branch-count ablation evaluates HorizontalRes, MatrixRes, and MatrixResGated on PROTEINS and DD for \(B=1,\ldots,8\). The sensitivity scan evaluates MatrixRes and MatrixResGated on fold 0 at \(B=3\), varying learning rate, dropout, hidden dimension, sparsity strength, and gate initialization where applicable. Because this sensitivity scan uses a single fold, it is interpreted as a search over promising operating regions rather than as final hyperparameter evidence. We therefore additionally rerun selected MatrixResGated candidates across all five folds: `sparse_lambda=0.02` on PROTEINS, and `lr=0.001`, `dim=32`, and `gate_init=0.2` on DD.

### 4.3 Mechanism metrics

We use three mechanism indicators in the main analysis:

- Branch diversity: mean pairwise distance between branch representations.
- Branch cosine similarity: mean cosine similarity between branch representations.
- Mean gradient norm: average gradient magnitude across tracked layers.

These metrics support the central interpretation. Branch diversity measures whether more branches create representational spread. Branch cosine similarity measures redundancy. Mean gradient norm tracks optimization weakening as residual topology and branch count become more complex. Additional diagnostics, including CKA, residual traffic totals, active ratios, and gate statistics, are retained for appendix-level support.

## 5. Results

### 5.1 Main benchmark comparison

Table 1 reports the default \(B=3\) benchmark. Residual topology changes performance, but its effect is dataset-dependent. On PROTEINS, HorizontalRes gives the strongest mean accuracy at \(0.7062 \pm 0.0323\), narrowly ahead of Plain at \(0.7044 \pm 0.0387\), MatrixRes at \(0.7035 \pm 0.0247\), and MatrixResGated at \(0.7026 \pm 0.0238\). VerticalRes is lower at \(0.6981 \pm 0.0402\).

On DD, VerticalRes performs best at \(0.7249 \pm 0.0247\). MatrixResGated is second at \(0.7181 \pm 0.0259\), followed by MatrixRes at \(0.7122 \pm 0.0362\), HorizontalRes at \(0.7113 \pm 0.0409\), and Plain at \(0.7053 \pm 0.0517\). This indicates that DD benefits from residual reuse, but the simplest depth-wise path is already highly competitive at the default branch budget.

On ENZYMES, MatrixResGated obtains the best mean accuracy at \(0.2933 \pm 0.0470\), followed by MatrixRes at \(0.2750 \pm 0.0577\), Plain at \(0.2683 \pm 0.0343\), HorizontalRes at \(0.2633 \pm 0.0584\), and VerticalRes at \(0.2550 \pm 0.0455\). The absolute scores are lower and more variable, so these results should be interpreted as evidence of a promising trend rather than a final claim of dominance.

**Table 1. Main benchmark results with GCNConv and \(B=3\).**

| Dataset | Plain | VerticalRes | HorizontalRes | MatrixRes | MatrixResGated |
|---|---:|---:|---:|---:|---:|
| PROTEINS | 0.7044 +/- 0.0387 | 0.6981 +/- 0.0402 | **0.7062 +/- 0.0323** | 0.7035 +/- 0.0247 | 0.7026 +/- 0.0238 |
| DD | 0.7053 +/- 0.0517 | **0.7249 +/- 0.0247** | 0.7113 +/- 0.0409 | 0.7122 +/- 0.0362 | 0.7181 +/- 0.0259 |
| ENZYMES | 0.2683 +/- 0.0343 | 0.2550 +/- 0.0455 | 0.2633 +/- 0.0584 | 0.2750 +/- 0.0577 | **0.2933 +/- 0.0470** |

Figure 1 visualizes the same benchmark ranking across datasets.

### 5.2 Branch-count ablation

Branch count has a non-monotonic effect. On PROTEINS, HorizontalRes achieves its best result at \(B=4\), reaching \(0.7125 \pm 0.0286\). MatrixRes peaks later at \(B=6\), reaching \(0.7062 \pm 0.0210\). MatrixResGated peaks earlier at \(B=3\), reaching \(0.7098 \pm 0.0219\). These results suggest that PROTEINS benefits from moderate or moderately large branch budgets, but the best scale depends on how residual exchange is constrained.

On DD, the preferred branch budget is smaller. HorizontalRes and MatrixRes both peak at \(B=2\), with \(0.7275 \pm 0.0304\) and \(0.7206 \pm 0.0411\), respectively. MatrixResGated performs best at \(B=1\), with \(0.7283 \pm 0.0244\). Additional branches do not consistently improve accuracy and often reduce it, indicating that DD is more sensitive to redundant branch expansion.

**Table 2. Best branch-count setting for each topology in the ablation.**

| Dataset | Model | Best \(B\) | Accuracy |
|---|---|---:|---:|
| PROTEINS | HorizontalRes | 4 | 0.7125 +/- 0.0286 |
| PROTEINS | MatrixRes | 6 | 0.7062 +/- 0.0210 |
| PROTEINS | MatrixResGated | 3 | 0.7098 +/- 0.0219 |
| DD | HorizontalRes | 2 | 0.7275 +/- 0.0304 |
| DD | MatrixRes | 2 | 0.7206 +/- 0.0411 |
| DD | MatrixResGated | 1 | 0.7283 +/- 0.0244 |

Figure 2 shows the full branch-count curves. The key pattern is not that a larger branch budget is better, but that each dataset and residual topology has a useful operating region.

### 5.3 MatrixResGated sensitivity

The fold-0 sensitivity scan shows that MatrixResGated is tunable, but its preferred operating region differs between datasets. On PROTEINS, the baseline MatrixResGated configuration reaches \(0.6726\). Mild sparsification with \(\lambda=0.02\) improves accuracy to \(0.6951\), while stronger sparsification with \(\lambda=0.1\) reduces it to \(0.6278\). Learning-rate and dimension changes do not improve over the baseline, and a lower gate initialization gives only a small change.

On DD, the baseline MatrixResGated score is \(0.7300\). More conservative settings improve the fold-0 result: learning rate \(0.001\), hidden dimension \(32\), and gate initialization \(0.2\) each reach \(0.7511\). In contrast, sparsity does not help, with \(\lambda=0.01\), \(0.02\), and \(0.1\) all below the baseline. This suggests that DD benefits from controlled matrix reuse, but prefers smaller or more conservative configurations rather than additional sparsification.

**Table 3. Strongest MatrixResGated fold-0 sensitivity settings.**

| Dataset | Setting | Accuracy | Interpretation |
|---|---|---:|---|
| PROTEINS | baseline | 0.6726 | default controlled matrix reuse |
| PROTEINS | `sparse_lambda=0.02` | 0.6951 | mild sparsification improves fold-0 accuracy |
| PROTEINS | `sparse_lambda=0.1` | 0.6278 | strong sparsification is harmful |
| DD | baseline | 0.7300 | default controlled matrix reuse |
| DD | `lr=0.001` | 0.7511 | conservative optimization helps |
| DD | `dim=32` | 0.7511 | smaller hidden dimension helps |
| DD | `gate_init=0.2` | 0.7511 | lower initial gate helps |

Figure 3 summarizes these sensitivity slices.

### 5.4 Five-fold check of tuned MatrixResGated candidates

The fold-0 sensitivity scan identifies promising settings, but it can overstate improvements if a setting matches one fold particularly well. To separate candidate discovery from evaluation, we reran the strongest MatrixResGated candidates across all five folds. Table 4 reports these follow-up checks.

On DD, the smaller hidden dimension is the most useful tuned candidate. `dim=32` reaches \(0.7215 \pm 0.0330\), improving over the default MatrixResGated main benchmark result of \(0.7181 \pm 0.0259\) while using fewer parameters. The lower learning rate candidate is essentially tied with the default at \(0.7181 \pm 0.0336\), and `gate_init=0.2` is lower at \(0.7131 \pm 0.0213\). Thus, the full-fold evidence supports the conservative-capacity interpretation for DD, but does not support all fold-0 improvements equally.

On PROTEINS, `sparse_lambda=0.02` reaches \(0.6909 \pm 0.0447\), below the default MatrixResGated main benchmark result of \(0.7026 \pm 0.0238\). This does not contradict the fold-0 sensitivity result; rather, it shows that the mild-sparsity gain was not stable across folds in the current run. The conservative conclusion is that sparsification remains a plausible control mechanism, but the current PROTEINS evidence is not strong enough to claim a five-fold improvement.

**Table 4. Five-fold checks of selected MatrixResGated candidates.**

| Candidate | Dataset | Accuracy | Mean loss | Parameters |
|---|---|---:|---:|---:|
| `PROTEINS_sparse_lambda_0.02` | PROTEINS | 0.6909 +/- 0.0447 | 0.6104 | 38,339 |
| `DD_lr_0.001` | DD | 0.7181 +/- 0.0336 | 0.5737 | 42,371 |
| `DD_dim_32` | DD | **0.7215 +/- 0.0330** | 0.5717 | 15,043 |
| `DD_gate_init_0.2` | DD | 0.7131 +/- 0.0213 | 0.5820 | 42,371 |

These results sharpen the role of the sensitivity scan. Single-fold scans are useful for identifying operating regions, but full-fold reruns are necessary before turning a candidate into a performance claim.

### 5.5 Mechanism analysis

The mechanism summaries support a common explanation for the rise-then-fall trend. Increasing branch count initially creates useful diversity: branches become less identical and can cover complementary feature subspaces. Past the useful regime, however, diversity alone is not sufficient. Residual traffic grows quickly, branch similarity may remain high, and gradient norms can weaken. The model becomes larger but not proportionally more useful.

For HorizontalRes on PROTEINS, moving from \(B=1\) to \(B=4\) increases branch diversity from \(0.0000\) to \(1.1450\), while branch cosine decreases from \(1.0000\) to \(0.8417\). This corresponds to the main accuracy improvement. Beyond \(B=4\), diversity continues increasing, reaching \(1.5441\) at \(B=8\), and branch cosine falls to \(0.7450\), but accuracy declines to \(0.6881\). The added branches are therefore not automatically beneficial.

MatrixRes shows a similar trend at a larger branch budget. On PROTEINS, its best result appears at \(B=6\), where branch diversity reaches \(3.7755\). The matrix topology can exploit more branches than HorizontalRes, likely because the two-dimensional stencil creates more reuse opportunities. Still, performance saturates once the branch budget becomes too large.

DD is more conservative. For HorizontalRes and MatrixRes, moving from \(B=1\) to \(B=2\) improves accuracy, but larger settings quickly lose value. MatrixRes at \(B=3\) still has branch cosine \(0.9875\), indicating that branches remain highly redundant even after the branch budget increases. In this case, additional branches add residual complexity without enough functional separation.

Figure 4 aligns the accuracy, branch diversity, branch cosine, and gradient-norm trends.

## 6. Discussion

### 6.1 Residual topology is dataset-dependent

The main benchmark argues against a universal residual topology. On PROTEINS, horizontal branch exchange is strongest at the default branch count. On DD, vertical residual reuse is strongest. On ENZYMES, the sparse/gated matrix variant is strongest, though variance is high. These outcomes suggest that the useful residual neighborhood depends on the dataset's tolerance for branch specialization and residual traffic.

This does not mean that matrix residuals are ineffective. Rather, matrix residuals appear more sensitive to branch budget and control. In the branch ablation, MatrixRes reaches its best PROTEINS result at a larger branch count than HorizontalRes, suggesting that the local two-dimensional stencil can exploit additional branches when the dataset supports them. MatrixResGated, by contrast, peaks earlier, which is consistent with a selective residual mechanism that avoids over-expansion.

### 6.2 Branch count is not a capacity knob alone

The branch-count results show that \(B\) should not be treated as a simple capacity knob. Increasing \(B\) changes the residual graph itself: it adds branches, residual paths, residual traffic, and optimization interactions. If the added branches separate into useful functional subspaces, accuracy can improve. If they remain redundant or weaken optimization, performance can decline even as representational diversity increases.

This distinction matters for multi-branch GNN design. A larger branch budget should be justified by evidence that branches are becoming usefully distinct, not merely by the fact that the model has more parameters. Branch diversity and cosine similarity provide a compact diagnostic for this question.

### 6.3 Controlled residual reuse is useful but not universal

The MatrixResGated sensitivity scan suggests that residual control has value, but the best form of control differs across datasets. PROTEINS shows a fold-0 benefit from mild sparsification, while DD shows fold-0 benefits from conservative optimization, smaller hidden dimension, and lower initial gate. The five-fold tuned-candidate reruns refine this picture: on DD, the smaller hidden dimension remains beneficial and parameter-efficient, whereas the other conservative settings do not consistently improve over the default. On PROTEINS, mild sparsification does not hold its fold-0 advantage across all folds.

The conservative interpretation is that controlled matrix reuse defines a tunable family rather than a plug-and-play winner. Candidate settings should be discovered with sensitivity scans and then validated across folds before being treated as improvements. The current evidence supports a smaller controlled matrix model on DD, while leaving PROTEINS sparsification as a promising but not yet stable direction.

### 6.4 Practical recommendations

For small or redundancy-prone graph-classification settings, start with a small branch budget and compare vertical residual reuse against controlled matrix reuse. For datasets where branches separate more readily, test horizontal and matrix residuals over a moderate branch range rather than fixing \(B=3\). In both cases, inspect branch diversity and branch cosine alongside accuracy. Accuracy alone reveals the endpoint, but the mechanism metrics explain whether branch expansion is producing useful specialization or redundant traffic.

## 7. Limitations

The current main benchmark uses GCNConv only. This keeps the comparison controlled but leaves open whether the same residual-topology trends hold for other propagation operators such as GINConv or GraphSAGE. The sensitivity scan is first-batch and fold-0 only, so it should be treated as evidence about promising operating regions rather than final hyperparameter selection. Although selected candidates were rerun across five folds, that follow-up is still limited to MatrixResGated on PROTEINS and DD. ENZYMES results are also variable and should be strengthened before making strong dataset-specific claims.

The experiments focus on three TUDataset benchmarks. This is appropriate for an initial graph-classification study, but broader datasets are needed to understand whether the same branch-count dynamics hold on larger or more diverse graph collections. Finally, the current mechanism analysis is correlational. It supports a coherent explanation, but it does not by itself prove that branch diversity, cosine similarity, or gradient norms causally determine the observed accuracy changes.

## 8. Conclusion

This paper studies residual connectivity in graph classification as a structured branch-by-layer design problem. The experiments show that residual topology matters, but its value depends on branch budget and dataset characteristics. Branch expansion helps when it creates useful representational diversity, then saturates or declines when residual complexity and redundancy dominate. Matrix-style residual reuse is most useful when paired with appropriate control, and its best settings differ across datasets.

The main implication is that residual design should be evaluated jointly with branch count and residual filtering. Future work should extend the comparison to additional graph operators, expand tuned-candidate validation beyond the current MatrixResGated checks, evaluate larger graph-classification datasets, and test whether adaptive residual-neighborhood selection can replace manually chosen vertical, horizontal, or matrix stencils.

## Figure Captions

**Figure 1. Main benchmark comparison.** Mean best test accuracy across five folds on PROTEINS, DD, and ENZYMES using a shared GCNConv backbone and default branch count \(B=3\). Residual topology changes the model ranking in a dataset-dependent way: HorizontalRes is strongest on PROTEINS, VerticalRes is strongest on DD, and MatrixResGated is strongest on ENZYMES. Error bars denote fold-level standard deviation.

**Figure 2. Branch-count ablation.** Effect of branch count \(B\) on HorizontalRes, MatrixRes, and MatrixResGated for PROTEINS and DD. Accuracy follows a rise-then-fall pattern rather than improving monotonically with more branches. PROTEINS tolerates a moderate-to-larger branch budget, while DD prefers smaller budgets.

**Figure 3. MatrixResGated sensitivity.** First-batch fold-0 sensitivity scan for MatrixResGated at \(B=3\). PROTEINS benefits most from mild sparsification, with `sparse_lambda=0.02` improving accuracy over the default setting. DD favors more conservative configurations, with lower learning rate, smaller hidden dimension, and lower gate initialization performing best.

**Figure 4. Mechanism branch dynamics.** Mechanism summary linking branch-count accuracy trends to branch diversity, branch cosine similarity, and mean gradient norm. Additional branches initially create useful diversity and reduce branch redundancy, but larger branch budgets can increase residual complexity without improving optimization.

## Appendix A. Result Sources

The manuscript uses the following summary files as the source of truth:

- `records/LATEST/summaries/benchmark_summary.csv`
- `records/LATEST/summaries/branch_ablation_summary.csv`
- `records/LATEST/summaries/parameter_sensitivity_summary.csv`
- `records/LATEST/summaries/tuned_candidate_summary.csv`
- `records/LATEST/summaries/mechanism_compact_summary.csv`

The figure files are:

- `figures/exp/fig_main_benchmark_gcnconv.pdf`
- `figures/exp/fig_branch_count_ablation.pdf`
- `figures/exp/fig_matrixresgated_sensitivity.pdf`
- `figures/exp/fig_mechanism_branch_dynamics.pdf`

## Appendix B. Appendix Material to Include in a Submission

The following diagnostics should be moved into supplementary tables or figures when converting this draft to a venue template:

- Fold-level benchmark rows from `records/LATEST/summaries/benchmark_fold_rows.csv`.
- Full branch-ablation rows from `records/LATEST/summaries/branch_ablation_summary.csv`.
- Full sensitivity rows from `records/LATEST/summaries/parameter_sensitivity_summary.csv`.
- CKA summaries from `records/LATEST/mechanism_summaries/cka_branch_summary.csv` and `records/LATEST/mechanism_summaries/cka_depth_summary.csv`.
- Residual traffic summaries from `records/LATEST/mechanism_summaries/residual_stats_summary.csv`.

## References

The citation keys in this Markdown draft are defined in `paper/references.bib`.
