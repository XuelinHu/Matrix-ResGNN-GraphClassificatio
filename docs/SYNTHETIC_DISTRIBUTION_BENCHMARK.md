# Synthetic Distribution Benchmark

This document tracks the distribution-controlled synthetic graph benchmark added for the Matrix-ResGNN paper.

## Datasets

| Dataset | Distribution | Label rule |
|---|---|---|
| `SYN_ER` | Erdos-Renyi `G(n,p)` | low vs high Bernoulli edge probability |
| `SYN_BA` | Barabasi-Albert scale-free graph | low vs high attachment count |
| `SYN_SBM` | two-block stochastic block model | weak vs strong community separation |
| `SYN_WS` | Watts-Strogatz small-world graph | low vs high rewiring probability |
| `SYN_REGULAR` | randomly permuted regular ring lattice | low vs high fixed degree |

Node features do not directly encode labels. The first feature is normalized node degree; the remaining dimensions are label-independent random features.

## Multi-Class Scaling Datasets

The multi-class extension keeps the same five graph distributions and creates class-count variants from 2 to 8 classes:

- `SYN_ER_C2` ... `SYN_ER_C8`
- `SYN_BA_C2` ... `SYN_BA_C8`
- `SYN_SBM_C2` ... `SYN_SBM_C8`
- `SYN_WS_C2` ... `SYN_WS_C8`
- `SYN_REGULAR_C2` ... `SYN_REGULAR_C8`

Each class is represented by a monotonic structural parameter:

| Distribution | Multi-class rule |
|---|---|
| `ER` | increasing Bernoulli edge probability |
| `BA` | increasing attachment count |
| `SBM` | increasing community separation |
| `WS` | increasing rewiring probability |
| `REGULAR` | increasing fixed degree |

The benchmark additionally reports macro-F1 and normalized accuracy:

```text
normalized_acc = (acc - 1 / num_classes) / (1 - 1 / num_classes)
```

This normalization makes 2-class and 8-class tasks comparable after removing the random-guess baseline.

## Smoke Run

Smoke data are intentionally small and are used only to verify the pipeline:

```bash
bash scripts/start_synthetic_smoke_background.sh
```

Scope:

- datasets: 5 synthetic datasets
- models: Plain, VerticalRes, HorizontalRes, MatrixRes, MatrixResGated
- operator: GCNConv
- folds: fold 0 only
- epochs: 3
- version: `SYN_SMOKE`

Progress files:

- `records/SYN_SMOKE/queue/synthetic_smoke.pid`
- `records/SYN_SMOKE/queue/synthetic_smoke.nohup.log`
- `records/SYN_SMOKE/queue/synthetic_smoke_5dist_gcn_fold0_status.json`
- `records/SYN_SMOKE/queue/synthetic_smoke_5dist_gcn_fold0_events.jsonl`

## Full Run

The full run follows the existing benchmark style across all comparison models, operators, and five folds:

```bash
bash scripts/start_synthetic_full_background.sh
```

Scope:

- datasets: 5 synthetic datasets
- models: 5 residual-topology families
- operators: GCNConv, GATConv, SAGEConv, GINConv
- folds: 0..4
- version: `SYN_FULL`

Progress files:

- `records/SYN_FULL/queue/synthetic_full.pid`
- `records/SYN_FULL/queue/synthetic_full.nohup.log`
- `records/SYN_FULL/queue/synthetic_full_5dist_4ops_5folds_status.json`
- `records/SYN_FULL/queue/synthetic_full_5dist_4ops_5folds_events.jsonl`

## Multi-Class Smoke Run

```bash
bash scripts/start_synthetic_multiclass_smoke_background.sh
```

Scope:

- datasets: 35 synthetic datasets, 5 distributions x class counts `C2` to `C8`
- models: Plain, VerticalRes, HorizontalRes, MatrixRes, MatrixResGated
- operator: GCNConv
- folds: fold 0 only
- epochs: 3
- version: `SYN_MULTI_SMOKE`

Progress files:

- `records/SYN_MULTI_SMOKE/queue/synthetic_multiclass_smoke.pid`
- `records/SYN_MULTI_SMOKE/queue/synthetic_multiclass_smoke.nohup.log`
- `records/SYN_MULTI_SMOKE/queue/synthetic_multiclass_smoke_5dist_c2c8_gcn_fold0_status.json`
- `records/SYN_MULTI_SMOKE/queue/synthetic_multiclass_smoke_5dist_c2c8_gcn_fold0_events.jsonl`

## Multi-Class Full Run

```bash
bash scripts/start_synthetic_multiclass_full_background.sh
```

Scope:

- datasets: 35 synthetic datasets, 5 distributions x class counts `C2` to `C8`
- models: 5 residual-topology families
- operators: GCNConv, GATConv, SAGEConv, GINConv
- folds: 0..4
- graphs per class: 200
- expected jobs: 3500
- version: `SYN_MULTI_FULL`
