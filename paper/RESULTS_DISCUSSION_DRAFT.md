# Results and Discussion Draft

## Branch-count ablation

We next examined how branch count changes the behavior of horizontal and matrix-style residual reuse on `PROTEINS` and `DD`. The effect of branch count was not monotonic. Instead, all three multi-branch families showed an initial gain region followed by a saturation or decline region, indicating that a larger branch budget is useful only when it creates productive representational diversity without overwhelming optimization.

On `PROTEINS`, `HorizontalRes` achieved its best result at `B=4` (`0.7125 +/- 0.0286`), `MatrixRes` peaked later at `B=6` (`0.7062 +/- 0.0210`), and `MatrixResGated` peaked earlier at `B=3` (`0.7098 +/- 0.0219`). These trends suggest that the protein benchmark benefits from moderate or moderately large branch budgets, but the optimal scale depends on how aggressively residual exchanges are constrained. The ungated matrix topology continued to benefit from additional branches longer than the purely horizontal variant, whereas the gated matrix model favored a smaller and more selective regime.

On `DD`, the preferred branch budget was smaller. `HorizontalRes` and `MatrixRes` both peaked at `B=2` (`0.7275 +/- 0.0304` and `0.7206 +/- 0.0411`, respectively), while `MatrixResGated` performed best already at `B=1` (`0.7283 +/- 0.0244`). After these points, additional branches did not improve accuracy and often reduced it. This indicates that the dense graph benchmark is more sensitive to redundant branch expansion and benefits more from limited reuse than from larger branch ensembles.

## Why branch count first helps and then hurts

The mechanism summaries suggest a common explanation for the rise-then-fall trend. As branch count increases from a minimal setting, branch diversity increases and the model gains access to a richer residual reuse pattern. However, once branch count moves beyond the useful regime, diversity continues to grow while optimization becomes weaker and residual traffic grows quickly. The result is a larger but less efficient multi-branch system.

For `HorizontalRes` on `PROTEINS`, the rise from `B=1` to `B=4` coincides with branch diversity increasing from `0.0000` to `1.1450`, while branch cosine decreases from `1.0000` to `0.8417`. This indicates that the additional branches are no longer exact replicas and begin to cover complementary subspaces. However, beyond `B=4`, diversity keeps increasing, yet accuracy declines. At `B=8`, branch diversity reaches `1.5441` and branch cosine drops further to `0.7450`, but test accuracy falls to `0.6881`. In parallel, the mean gradient norm becomes smaller than in the low-branch regime. This pattern suggests that the model crosses from helpful specialization into over-fragmentation.

`MatrixRes` shows the same pattern at a larger scale. On `PROTEINS`, its best result arrives only at `B=6`, by which point branch diversity has grown to `3.7755`. The matrix topology can therefore exploit a larger branch budget than `HorizontalRes`, presumably because its local two-dimensional reuse pattern makes more use of the extra branches. Still, performance stops improving once the branch budget becomes too large. After `B=6`, diversity continues to rise or remains high, but accuracy no longer benefits. This suggests that a larger residual neighborhood can delay saturation, but it does not remove it.

The `DD` results are even more conservative. For both `HorizontalRes` and `MatrixRes`, moving from `B=1` to `B=2` improves accuracy, but larger values quickly become counterproductive. Importantly, branch diversity does increase with `B`, yet branch cosine remains extremely high in the early multi-branch regime. For example, `MatrixRes` at `B=3` still has branch cosine `0.9875`, even though its diversity is already much larger than at `B=2`. This means that the added branches do not separate into sufficiently distinct functional subspaces to justify the extra residual complexity. In this setting, additional branches mainly increase capacity and traffic without providing enough new information.

The gated matrix variant supports this interpretation. `MatrixResGated` generally peaks earlier than `MatrixRes`, and its active ratio remains lower across comparable branch settings. On `PROTEINS`, this helps the model stop before over-expansion becomes too harmful, which is why the best result appears at `B=3` rather than `B=6`. On `DD`, the same selectivity makes the smallest settings most attractive. In short, branch count helps when it creates usable diversity, and hurts when residual complexity grows faster than optimization quality or functional separation.

## Parameter sensitivity of MatrixResGated

We then examined the tunable region of `MatrixResGated` using first-batch sensitivity scans on `fold0`, `B=3`, and `GCNConv`. The results show that the controlled matrix model is tunable, but its preferred configuration differs substantially between `PROTEINS` and `DD`.

On `PROTEINS`, the baseline configuration achieved `0.6726`, and the clearest improvement came from mild sparsification. Setting `sparse_lambda=0.02` increased performance to `0.6951`, whereas stronger sparsification (`0.1`) reduced it to `0.6278`. Learning-rate and dimensional changes were less helpful: `lr=0.001` was too conservative, and both `dim=32` and `dim=128` underperformed the baseline `dim=64`. Gate initialization also showed a mild preference for a lower starting gate, with `gate_init=0.2` slightly outperforming the default. Taken together, the protein benchmark appears to favor a moderately expressive gated matrix model with light sparsity rather than aggressive pruning.

On `DD`, the preferred region was lighter and more conservative. The baseline score was `0.7300`, but `lr=0.001`, `dim=32`, and `gate_init=0.2` all reached `0.7511`. In contrast, the sparsity sweep did not help: even `sparse_lambda=0.01` underperformed the baseline, and larger penalties were no better. This pattern indicates that `DD` benefits from controlled reuse, but not from strong post hoc sparsification once gating is already present. A smaller hidden size, smaller learning rate, and lower initial gate appear to keep the matrix model in a more stable operating region.

These results imply that `MatrixResGated` does not have a single universal optimum. Instead, the usable region depends on the dataset’s tolerance for branch specialization and residual traffic. `PROTEINS` supports a moderately expressive setting with mild sparsity, whereas `DD` favors lighter and more conservative configurations.

## Recommended mechanism indicators for the main paper

The mechanism analysis currently contains several candidate signals, but not all of them are equally necessary for the main text. To keep the paper readable, the main narrative should focus on three indicators:

- `branch diversity`
- `branch cosine similarity`
- `mean gradient norm`

These three already support the central explanation:

- branch diversity captures whether added branches create distinct representations
- branch cosine shows whether those branches remain overly redundant
- mean gradient norm tracks the optimization weakening that appears as branch count grows

The remaining indicators, especially `CKA`, can be retained as supporting or appendix material. CKA is useful as a confirmatory representation-alignment view, but it is not needed as a first-line explanation if diversity and cosine already capture the redundancy pattern. The same applies to residual traffic totals and active ratios: they are helpful for validating the interpretation, but they should play a secondary role in the main figures and discussion.
