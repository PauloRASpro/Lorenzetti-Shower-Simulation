# Experimental result analyses

Each of the three article notebooks runs the original experimental plotting functions and the original inline result-figure statements with the corresponding saved event predictions, regional tables and measured timing metadata.

## Experimental inputs

- `Data/Lorenzetti_electrons_merge_root.csv` (or the supported compressed archive): pre-training calorimetric distributions and the 100-ring input variables.
- `artifacts/pickles/`: historical `*_noSEG_Full_DF`, `*_noSEG_performance_metrics`, `*_noSEG_cv_metrics_df`, `integrated_result_*` and associated archived results.
- `artifacts/metadata/prediction_times.csv`: optional numerical prediction times extracted from trusted training-time archives for timing-based performance figures.
- `artifacts/new_runs/<article>/<run_id>/`: individually saved model training runs, including replayable outputs.

## Result coverage

The original experimental notebook's result section starts at **Comparison Between All Methods**. The inventory of its scientific plotting cells is in [ORIGINAL_RESULT_OUTPUT_INVENTORY.csv](ORIGINAL_RESULT_OUTPUT_INVENTORY.csv). The article notebooks execute the journal-figure set, the model and representation comparison plots, ranking views, top-ten residual distributions, regional curves, resolution curves, paired Q–Q tests and supplementary result tables. One-off function-development tests and intermediate `.head()` displays are not part of this publication-facing analysis.

The journal figures are indexed by article and figure number in [PAPER_FIGURE_INDEX.md](PAPER_FIGURE_INDEX.md). Their generated outputs, and any unavailable inputs or incomplete panels, are recorded in `paper_figure_coverage.csv` and `paper_figure_coverage.json` under `results/replayed/<article>/paper_figures/`. Supplementary figures and tables are in `additional_result_figures/`. Each rendered PNG has a companion JSON record of figure size, DPI, axes, fonts, legends and the calling source line.

Predictions, physical energies, fitted representation dimensions and measured inference times are read from the experiment artifacts. Timing-based figures require recorded numerical timing information, and `D0` additionally requires the actual feature counts. No inference times are guessed when a historical estimator pickle cannot be deserialized.

## Graphical environment

The notebooks use the source plotting functions and Matplotlib setup, including the original TeX font preamble. To obtain source-matched typography, install a LaTeX distribution with `lmodern`, `amsmath` and `dvipng`. Computer-specific Matplotlib/LaTeX versions can still affect rasterization and text extents.
