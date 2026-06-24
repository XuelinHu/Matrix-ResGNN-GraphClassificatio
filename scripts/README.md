# Script Inventory

Use the Conda environment `pyg` for Python execution:

```powershell
conda run -n pyg python scripts/<script_name>.py
```

## Experiment runners

- `run_single.py`: run one dataset/model/operator/fold configuration.
- `run_benchmark.py`: run the main real-data benchmark queue.
- `run_missing_benchmark_queue.py`: fill missing benchmark jobs from an expected queue.
- `run_branch_ablation.py`: run branch-count ablations.
- `run_parameter_sensitivity.py`: run fold-0 parameter sensitivity scans.
- `run_tuned_candidates.py`: rerun selected tuned MatrixResGated candidates.

Shell launchers:

- `run_synthetic_smoke.sh`
- `run_synthetic_full.sh`
- `run_synthetic_multiclass_smoke.sh`
- `run_synthetic_multiclass_full.sh`
- `start_synthetic_smoke_background.sh`
- `start_synthetic_full_background.sh`
- `start_synthetic_multiclass_smoke_background.sh`
- `start_synthetic_multiclass_full_background.sh`

## Summary scripts

- `summarize_benchmark.py`: summarize the main benchmark.
- `summarize_branch_ablation.py`: summarize branch-count ablations.
- `summarize_parameter_sensitivity.py`: summarize sensitivity scans.
- `summarize_tuned_candidates.py`: summarize tuned-candidate reruns.
- `summarize_mechanism_artifacts.py`: summarize raw mechanism artifacts.
- `summarize_mechanism_compact.py`: create the compact mechanism table used by the paper.

## Figure scripts

- `generate_suite_figures.py`: generate the main benchmark and model-win figures.
- `generate_ablation_figures.py`: generate branch-count ablation figures.
- `generate_mechanism_figure.py`: generate mechanism summary figures.
- `generate_synthetic_multiclass_figures.py`: generate synthetic multi-class figures.
- `generate_synthetic_graphs.py`: generate synthetic graph datasets used by the synthetic benchmark.
- `plot_style.py`: shared plotting style helper.

## Export and checks

- `export_result_csvs_to_excel.py`: export selected CSV summaries to Excel.
- `export_residual_analysis.py`: export residual-analysis tables.
- `check_benchmark_completeness.py`: check expected benchmark completion.

For submission-facing artifacts, prefer the frozen outputs under `paper/`, `figures/`, and `records/` rather than rerunning long experiments unless a new revision requires it.
