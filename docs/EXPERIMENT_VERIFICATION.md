# Experiment execution and acceptance criteria

## Repository inputs

- Original Lorenzetti events: `data/Lorenzetti_electrons_merge_root.csv` (or its project archive, extracted by the notebook). The input columns are `0`–`99`, `mc_match_et`, `cluster_truthet`, and `cluster_trutheta`.
- Historical saved experiment results: `artifacts/pickles/`, preserving original filenames. These files are read only when explicitly trusted.
- New experiments: uniquely named directories under `artifacts/new_runs/<study>/<run-id>/`. Previously saved results are not overwritten.

## Execution workflow

1. In the notebook configuration, use `EXECUTION_MODE='replay'`, `REPLAY_SOURCE='original'` for the historical saved predictions.
2. Use `EXECUTION_MODE='train'` to run the selected method/representation configurations with the Lorenzetti CSV. Newly fitted models, eight types of result DataFrames, and generated figures are saved in the new run directory.
3. Use `EXECUTION_MODE='replay'` and `REPLAY_SOURCE='latest_new_run'` (or `'specific_new_run'` with `NEW_RUN_ID`) to reproduce the analysis of a particular newly trained model.
4. Check `results/replayed/<study>/paper_figures/paper_figure_coverage.csv` for the exact status of every experimental article figure.

## Source-to-figure correspondence

See [PAPER_FIGURE_INDEX.md](PAPER_FIGURE_INDEX.md). The indexed experimental outputs comprise 12 figure groups for IEEE TNS, 16 for Measurement and 16 for Information Fusion. A group is marked `generated` only if **all required subgroups** have produced images. Figure images from the articles are never used as inputs. Original plotting statements and source-cell hashes are preserved in `src/lorenzetti_repro/original_article_cells.py`. Original plotting functions are in `original_plotting.py`. The Matplotlib source settings (`usetex=True`, LaTeX preamble, and Lorenzetti legend) are in `original_figure_environment.py`.

## Local verification

- Unit tests for splitting, metrics, representations, models, hybrid/stacked estimators, archive I/O, isolated new runs and figure source completeness: `python -m pytest -q`.
- Integration test: execute all code cells of each of the three publication notebooks in a Jupyter kernel with compatible, *synthetic* DataFrames and source-event CSV. Verify generated tables, PNG files, and the per-figure coverage file. Synthetic test data are not included in the research distribution.
- Rendering tests using the original LaTeX preamble have been performed for the original raw-data diagnostic figure groups. Full notebook integration testing may disable TeX **only within the automated synthetic testing environment** to limit the number of LaTeX subprocesses.

## Scientific completeness checks

For a fully data-backed reproduction, every required original file must be present. The optional fit-quality figure requires the original `distfit` dependency. Segmented IQR/MAE/RMSE and boxplot panels require the archived integrated regional DataFrames. The D0 trade-off requires measured prediction times *and* the original feature counts (including CORR/MUTI threshold representations). The complete per-figure coverage index reports missing inputs rather than replacing them with values from publication PDFs.

The historic per-event pickle collection and the original 302,619-event file are not bundled in this source-code archive. The source-code tests therefore cannot establish numerical or pixel-identical reproduction of published results without those inputs and a compatible library/font environment.
