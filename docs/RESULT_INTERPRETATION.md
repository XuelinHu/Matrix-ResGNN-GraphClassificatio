# Result Interpretation Notes

## 1. Why does branch count first help and then hurt?

The branch-count trend is not monotonic. Across `PROTEINS` and `DD`, the common pattern is:

- small branch growth improves performance because it creates useful branch-level diversity
- too many branches reduce performance because branch similarity remains high while optimization becomes weaker and the residual budget grows quickly

### PROTEINS

- `HorizontalRes` peaks at `B=4` with `0.7125`, then declines at `B>=5`
- `MatrixRes` peaks later at `B=6` with `0.7062`
- `MatrixResGated` peaks earlier at `B=3` with `0.7098`

Interpretation:

- On `PROTEINS`, moderate horizontal diversification is helpful.
- `HorizontalRes` improves from `B=1` to `B=4` while:
  - branch diversity rises from `0.0000` to `1.1450`
  - branch cosine drops from `1.0000` to `0.8417`
  - branch CKA drops from `1.0000` to `0.9797`
- After `B=4`, diversity keeps increasing, but the benefit saturates:
  - at `B=8`, diversity is `1.5441`, but accuracy falls to `0.6881`
  - branch cosine is already much lower at `0.7450`, suggesting stronger specialization but less stable cooperation
  - mean gradient norm also becomes smaller than the low-branch regime, indicating weaker optimization signal

- `MatrixRes` needs more branches before it benefits.
  - accuracy is flat from `B=2` to `B=5`
  - the best result arrives at `B=6`
  - meanwhile branch diversity becomes much larger than in `HorizontalRes`, reaching `3.7755` at `B=6`
- This suggests that matrix connectivity can exploit a larger branch budget, but only up to a point.
- Beyond `B=6`, additional branches keep increasing diversity but do not improve accuracy, so the extra connectivity starts acting more like noise or redundancy than useful reuse.

- `MatrixResGated` peaks earlier than ungated `MatrixRes`.
  - best at `B=3`
  - active ratio stays lower than `MatrixRes`
  - branch cosine declines more gently
- This supports the view that sparse control is useful when the matrix topology is already expressive. Once branch count is too high, gating can limit damage, but it cannot fully recover the loss from over-expansion.

### DD

- `HorizontalRes` peaks at `B=2` with `0.7275`
- `MatrixRes` peaks at `B=2` with `0.7206`
- `MatrixResGated` peaks at `B=1` with `0.7283`, then mostly declines

Interpretation:

- `DD` prefers a smaller branch budget than `PROTEINS`.
- From `B=1` to `B=2`, both `HorizontalRes` and `MatrixRes` gain from introducing limited branch interaction:
  - `HorizontalRes` branch diversity rises from `0.0000` to `0.1687`
  - `MatrixRes` branch diversity rises from `0.0000` to `0.1834`
  - accuracy improves in both cases
- After that point, diversity rises sharply, but accuracy does not.
  - for example, `MatrixRes` diversity reaches `1.2973` at `B=3` and `2.3278` at `B=8`
  - accuracy stays below the `B=2` level

- The key signal is that branch similarity remains very high even when branch count grows:
  - `MatrixRes` branch cosine is still `0.9875` at `B=3`
  - `MatrixResGated` branch cosine is `0.9732` at `B=3`
- So on `DD`, extra branches do not create enough genuinely distinct subspaces to justify the added residual complexity.

- `MatrixResGated` doing best at `B=1` indicates that the strongest gain on `DD` comes from controlled residual reuse rather than larger branch ensembles.

### Operational summary

- Increasing `B` helps first because it raises branch diversity and expands the reuse pattern.
- It hurts later because:
  - diversity continues to grow after the useful regime
  - branch similarity often remains high enough to indicate partial redundancy
  - gradient magnitude tends to weaken as the system becomes larger
  - residual traffic grows rapidly with branch count, especially for matrix topologies

- In short:
  - `PROTEINS` benefits from a moderate-to-larger branch budget
  - `DD` benefits from a smaller branch budget
  - `MatrixResGated` is more conservative and usually peaks earlier than ungated `MatrixRes`

## 2. Tunable region for MatrixResGated

The first-batch sensitivity scan was run on `fold0`, `B=3`, `GCNConv`.

### PROTEINS

Baseline:

- `0.6726`

Stable-to-use region:

- `dropout` around `0.1` to `0.5`
- `gate_init=0.2` is slightly better than the baseline gate initialization
- the only clearly beneficial sparse setting is `sparse_lambda=0.02`

Key observations:

- `sparse_lambda=0.02` gives the best result, `0.6951`
- `sparse_lambda=0.1` is clearly too strong and drops to `0.6278`
- `lr=0.001` is too conservative here
- `dim=32` and `dim=128` both underperform the baseline, so `dim=64` remains the safer choice

Interpretation:

- On `PROTEINS`, the gated matrix model is tunable but not extremely forgiving.
- Mild sparsification helps, but aggressive pruning hurts.
- The most defensible default region is:
  - `lr in [0.003, 0.005]`
  - `dropout in [0.1, 0.5]`
  - `gate_init around 0.2 to 0.8`
  - `sparse_lambda around 0.02`

### DD

Baseline:

- `0.7300`

Stable-to-use region:

- `lr=0.001` performs best
- `dim=32` performs best
- `gate_init=0.2` performs best
- sparse penalties larger than `0.01` hurt

Key observations:

- best values are tied around `0.7511` for:
  - `lr=0.001`
  - `dim=32`
  - `gate_init=0.2`
- `sparse_lambda=0.01` is already worse than the baseline, and larger penalties are not helpful

Interpretation:

- On `DD`, the gated matrix model prefers a lighter and more conservative configuration:
  - smaller learning rate
  - smaller hidden dimension
  - lower initial gate
  - weak or no additional sparsity pressure

- This suggests that `DD` benefits from controlled reuse, but not from aggressively thinning the residual channels after they are already sparse-gated.

### Practical default for the next round

If we need one paper-facing default for `MatrixResGated`, the current evidence supports:

- `PROTEINS`:
  - `lr=0.003`
  - `dropout=0.1 or 0.3`
  - `dim=64`
  - `gate_init=0.2`
  - `sparse_lambda=0.02`

- `DD`:
  - `lr=0.001`
  - `dropout=0.1 or 0.3`
  - `dim=32`
  - `gate_init=0.2`
  - `sparse_lambda=0.0 to 0.01`
