# Experiment inputs, saved results, and figure coverage

## Input dataset

The training dataset is `Lorenzetti_electrons_merge_root.csv`. Keep the original CSV or its compressed archive under `Data/` in the **local repository clone**. The executable notebooks discover either `Data/` or `data/` and can extract ZIP, GZip, TAR, 7z (with `py7zr`), or RAR (with `rarfile` and an installed extraction backend). The dataset contains 100 ring variables and the calorimeter variables `mc_match_et`, `cluster_truthet` and `cluster_trutheta`. A full benchmark has 302,619 reconstructed events.

Files available on the GitHub website are not automatically available in a local notebook session: they must be included when cloning or downloaded with Git LFS when appropriate. If the local source dataset is absent, the notebook raises an actionable error before reporting pre-training figures.

## Historical experiment outputs

Put trusted saved DataFrames under `artifacts/pickles/`, retaining original method and file names. The replay expects event-level `*_noSEG_Full_DF.pkl[.gz]` files, configuration-level `*_noSEG_performance_metrics.pkl[.gz]`, and integrated regional tables `integrated_result_*_noSEG_Full_DF.pkl[.gz]`. Where needed for model-complexity plots, numerical CPU times come from the saved timing records. Never load unknown pickle files.

## Execution modes

- Original experiment results: `EXECUTION_MODE='replay'`, `REPLAY_SOURCE='original'`.
- New model training: `EXECUTION_MODE='train'`. Outputs are written exclusively to a new `artifacts/new_runs/<article>/<run_id>/` directory, never to the historical archive.
- Previously saved new results: `EXECUTION_MODE='replay'`, with `REPLAY_SOURCE='latest_new_run'` or `REPLAY_SOURCE='specific_new_run'` and `NEW_RUN_ID`.

Every notebook runs the source-data diagnostics, its model-specific experimental results, and the associated figure/tables code. Figure geometry is controlled by the author-defined plotting functions and Matplotlib/LaTeX environment. An original result is considered available only when the required source events, saved predictions and numeric metadata are present.

## Analysis groups

For each article, the publication plots are indexed in `docs/PAPER_FIGURE_INDEX.md`. In addition, the original comparison block contains all-configuration residual histograms/boxplots, global `d0`/`D0` rankings, prediction times, regional calibration-factor plots, relative energy resolution, threshold acceptance, Q-Q analysis, inferential interval graphics and other experimental comparisons. The output directory contains the computed figures and tables, the source metadata for every plot, and the per-figure coverage file.

The input order for histograms and boxplots is decreasing calibrated-energy RMSE; plotting layouts, axis labels and legend properties remain defined by the original source routines. Region-integrated statistics use the original formulas from the source notebook and are recovered from the stored per-bin vectors only if their numerical columns are absent.

For matching LaTeX fonts, install the TeX packages `lmodern` and `amsmath`, plus `dvipng`. The same source code can otherwise render differently across operating systems and plotting-library versions.
