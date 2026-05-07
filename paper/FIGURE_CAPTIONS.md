# Figure Captions

## Figure 1. Main benchmark comparison

**File:** `figures/exp/fig_main_benchmark_gcnconv.pdf`

Mean best test accuracy across five folds on PROTEINS, DD, and ENZYMES using a shared GCNConv backbone and default branch count \(B=3\). Residual topology changes the model ranking in a dataset-dependent way: HorizontalRes is strongest on PROTEINS, VerticalRes is strongest on DD, and MatrixResGated is strongest on ENZYMES. Error bars denote fold-level standard deviation.

## Figure 2. Branch-count ablation

**File:** `figures/exp/fig_branch_count_ablation.pdf`

Effect of branch count \(B\) on HorizontalRes, MatrixRes, and MatrixResGated for PROTEINS and DD. Accuracy follows a rise-then-fall pattern rather than improving monotonically with more branches. PROTEINS tolerates a moderate-to-larger branch budget, with peaks at \(B=4\) for HorizontalRes, \(B=6\) for MatrixRes, and \(B=3\) for MatrixResGated. DD prefers smaller budgets, with the strongest results at \(B=2\) for HorizontalRes and MatrixRes and \(B=1\) for MatrixResGated.

## Figure 3. MatrixResGated sensitivity

**File:** `figures/exp/fig_matrixresgated_sensitivity.pdf`

First-batch fold-0 sensitivity scan for MatrixResGated at \(B=3\). PROTEINS benefits most from mild sparsification, with `sparse_lambda=0.02` improving accuracy over the default setting. DD favors more conservative configurations, with lower learning rate, smaller hidden dimension, and lower gate initialization performing best. The pattern indicates that controlled matrix residual reuse has dataset-specific operating regions.

## Figure 4. Mechanism branch dynamics

**File:** `figures/exp/fig_mechanism_branch_dynamics.pdf`

Mechanism summary linking branch-count accuracy trends to branch diversity, branch cosine similarity, and mean gradient norm. Additional branches initially create useful diversity and reduce branch redundancy, but larger branch budgets can increase residual complexity without improving optimization. This explains why branch count improves performance only up to a dataset- and topology-specific regime before saturating or declining.

