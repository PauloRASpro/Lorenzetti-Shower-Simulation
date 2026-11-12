# Computational reproducibility

The primary source code resides in `src/lorenzetti_repro/` and is organized by model, representation, training protocol, event-level metrics, statistical inference and visualization. Historical plotting function bodies are maintained in `original_plotting.py` and their source-cell mapping is recorded in `ORIGINAL_FUNCTION_MAP.json`.

Each notebook supports model fitting and archived-result analysis. The latter path consumes the saved `*_noSEG_Full_DF` predictions and the corresponding segmented, CV and performance DataFrames. Figures are computed from these inputs and recorded CPU timings when relevant.

The separation of D70 and D30, representation fitting, stage-specific nested cross-validation and final evaluation follow the thesis methodology. Model selection in Stages II–III is conditional on the D70-fitted representation rather than fully nested joint selection of both operators. The results concern the Lorenzetti benchmark and do not imply deployment readiness within the official ATLAS HLT.

A numerical equivalence check of a historical experiment requires the original event order, saved predictions, representation identifiers and compatible software dependencies. The generated `publication_figure_coverage.csv` and computed metrics provide traceability for each notebook execution.

## Replay sources and isolated new training

Set `EXECUTION_MODE='replay'` and `REPLAY_SOURCE='original'` to read `artifacts/pickles/`. For new Lorenzetti model fitting, set `EXECUTION_MODE='train'`. A unique timestamped directory `artifacts/new_runs/<article>/<run-id>/` will contain its saved predictions and integrated data in `pickles/`, fitted estimators in `trained_models/`, recorded parameters in `manifests/`, and generated graphics in `figures/`.

To analyze a completed new run later without refitting, use `EXECUTION_MODE='replay'` and `REPLAY_SOURCE='latest_new_run'`, or `REPLAY_SOURCE='specific_new_run'` with `NEW_RUN_ID` set to a listed run ID. New training cannot write into the historical pickle directory.

The original Lorenzetti input is `Lorenzetti_electrons_merge_root.csv` (processed source: `Lorenzetti_simulation_raw_20250815.csv`). Retain the compressed distribution in `data/` or `Data/`.
