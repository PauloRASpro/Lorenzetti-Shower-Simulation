# Code and result provenance

| Research stage | Model or computation | Executable module |
|---|---|---|
| IEEE TNS | Linear, polynomial, decision-tree and BDT regression | `src/lorenzetti_repro/model_runner.py`, `article_training.py` |
| IEEE TNS | B-spline regression | `src/lorenzetti_repro/models/bspl_original_cell513.py` |
| IEEE TNS | Segment-specific benchmark | `src/lorenzetti_repro/models/stage1_segmented.py` |
| All studies | Correlation, mutual information, PCA, ICA and 100-ring representation | `src/lorenzetti_repro/representations.py` |
| Measurement | Residual BDT–RF, RF–BDT and stacked ensembles | `src/lorenzetti_repro/models/hybrids.py` |
| Measurement/Information Fusion | Conditional nested validation and score selection | `src/lorenzetti_repro/selection.py` |
| Information Fusion | DNN with tabular inputs | `src/lorenzetti_repro/models/dnn_original_cell527.py` |
| Information Fusion | Region-specific graph and relational–tabular fusion | `src/lorenzetti_repro/models/gnn_original_cell562.py` |
| All studies | Loading saved DataFrames and models | `src/lorenzetti_repro/legacy_artifacts.py`, `article_workflow.py` |
| All studies | Metrics, paired statistics and segmented analyses | `metrics.py`, `statistics.py`, `analysis_tables.py`, `archive_tables.py` |
| All studies | Article figure generation through the original source plotting routines | `original_plotting.py`, `original_replay.py`, `original_inline_figures.py` |

### Source notebook artifact fields

The original experiment stored per-configuration predictions as `method`, `method-selection`, `useRings`, `ypred`, `yte`, `cluster_truthet`, `mc_match_et`, `cluster_trutheta` and `calib_et`. The notebooks load these fields and compute energy residuals, regional profiles, energy resolution and truth-centered EAR curves. The `result_*` / `integrated_result_*` artifacts hold aggregated lists of factor values by original intervals. Source articles are available as `papers/*.pdf`.

Numerical comparison tables in `results/reference/` are reference material and are not used as input to plotting. Generated results are written to `results/replayed/` and new model checkpoints to `artifacts/new_runs/`.
