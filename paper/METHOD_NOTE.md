# Method Note

## Core framing

The new paper will treat residual connectivity as a structured design problem on a two-dimensional `branch x layer` grid. The goal is not to propose another isolated skip variant, but to compare three constrained residual neighborhoods under one shared graph-classification backbone.

The paper-facing model family is:

- `Plain`
- `Vertical-Res`
- `Horizontal-Res`
- `Matrix-Res`
- `Matrix-Res (sparse/gated)`

## Unified notation

Let `H_b^(l)` denote the hidden state at branch `b` and layer `l`, where:

- `b in {1, ..., B}` indexes branches
- `l in {1, ..., L}` indexes message-passing depth

Let `phi_b^(l)(., A)` denote the base graph propagation operator at branch `b` and layer `l`, applied under graph structure `A`.

The branch-local propagation is

```text
Z_b^(l) = phi_b^(l)(H_b^(l-1), A).
```

Residual-enhanced propagation is written as

```text
H_b^(l) = Z_b^(l) + sum_{(b', l') in N(b, l)} R_{(b', l') -> (b, l)}(H_{b'}^(l')).
```

Here:

- `N(b, l)` is the allowed residual neighborhood for location `(b, l)`
- `R` is the residual transform on a source state before injection
- `R` is initially identity, and later extended to gated or sparse variants

This notation keeps the backbone, operator class, and classifier fixed while changing only the residual neighborhood and residual transform.

## Three residual neighborhoods

### Vertical-Res

Vertical residuals reuse the previous layer on the same branch:

```text
N(b, l) = {(b, l - 1)}.
```

This is the depth-wise reuse baseline.

### Horizontal-Res

Horizontal residuals exchange information across adjacent branches at the same depth:

```text
N(b, l) = {(b - 1, l), (b + 1, l)}.
```

Boundary branches use only valid neighbors. This isolates branch-wise exchange without explicit cross-depth mixing.

### Matrix-Res

Matrix residuals combine local vertical, horizontal, and diagonal reuse:

```text
N(b, l) = {
  (b, l - 1),
  (b - 1, l), (b + 1, l),
  (b - 1, l - 1), (b + 1, l - 1)
}.
```

This is a local two-dimensional stencil rather than arbitrary dense mixing. The design intentionally preserves structure: only adjacent branches and adjacent depths may contribute residual signals.

## Matrix-Res with sparse or gated control

The controlled variant keeps the same matrix neighborhood but replaces identity residual injection with a filtered transform:

```text
R(U) = alpha * U
```

for gated residuals, or

```text
R(U) = softshrink_lambda(U)
```

for sparse residuals.

The purpose of this variant is not to introduce another topology, but to test whether local two-dimensional reuse should be passed densely or selectively.

## Initial implementation assumptions

- branch count `B = 3`
- hidden depth `L = 4` as the initial default
- later ablation on `B = 1...8`
- same graph-classification benchmark family as the current repository baseline
- same training protocol across all five model variants

## Paper-level research questions

The new manuscript should be built around these questions:

1. Is same-branch vertical reuse sufficient, or does same-depth horizontal exchange provide different gains?
2. Does a local two-dimensional residual neighborhood outperform one-axis residual reuse under a fixed graph-classification protocol?
3. When matrix-style reuse is helpful, does it benefit from dense injection or from sparse/gated control?
4. How do residual topology and branch count affect stability, accuracy, and representation dynamics?
