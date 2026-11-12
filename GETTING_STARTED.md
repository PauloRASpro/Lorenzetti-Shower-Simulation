# How to run the publication notebooks

## Original dataset

The original experiment notebook loads `Lorenzetti_electrons_merge_root.csv` for the Lorenzetti benchmark. A processed CSV may also be named `Lorenzetti_simulation_raw_20250815.csv`. Keep the compressed dataset already distributed in this repository under `data/` (or `Data/`). `EXTRACT_DATA=True` extracts the archive. The study uses 302,619 events with 100 ring inputs, `cluster_truthet`, `mc_match_et`, and `cluster_trutheta`.

## Original figure typography

The original research notebook uses `matplotlib.rcParams["text.usetex"] = True`, the `lmodern` and `amsmath` LaTeX packages, and a Lorenzetti source label containing TeX. For matching original font rendering, install **LaTeX + dvipng + lmodern + amsmath** at the operating-system level, in addition to the Python dependencies. The paper notebooks enable these original settings automatically when the LaTeX tools are available and warn otherwise. The original figure source and the exact plotting-call configuration control dimensions, labels, font sizes, colors, legends and layout. Differences in Matplotlib/TeX/font versions can affect pixel-identical output.

## Stored published results — no training

Place your trusted archives (`*_noSEG_Full_DF.pkl`, `*_noSEG_performance_metrics.pkl`, etc.) in `artifacts/pickles/` and use `EXECUTION_MODE='replay'` with `REPLAY_SOURCE='original'`. Open the relevant paper notebook and choose **Restart & Run All**. Figures and tables are calculated from the archived predictions. Run the notebook to obtain `paper_figure_coverage.csv`: every indexed experimental figure is marked `generated`, `partial`, or `missing_input`; a missing panel is never marked as complete. The historical pickle files remain unchanged. Optional numerical prediction times can be extracted from model-containing archives when their historical class definitions are available. Failure to read such a model must not stop prediction-based analysis.

## Training a new run — no overwrite

Choose `EXECUTION_MODE='train'` and optionally specify `TRAIN_CONFIGURATIONS`. Training requires the original Lorenzetti dataset. Each run receives a unique timestamp/ID under `artifacts/new_runs/<article>/<id>/`. It writes new model prediction pickles to `pickles/`, estimators to `trained_models/`, metadata to `manifests/`, and graphs to `figures/`. The code refuses to write into `artifacts/pickles/` and refuses to create an existing output directory.

## Replaying a new training

Choose `EXECUTION_MODE='replay'` and `REPLAY_SOURCE='latest_new_run'` or `REPLAY_SOURCE='specific_new_run'` with `NEW_RUN_ID` set to a saved run folder. Existing runs are not retrained.

## Limitations and safety

Only unpickle data you trust; arbitrary pickle files can execute code. The availability of a plotting function does not guarantee a historically matching figure if the corresponding source prediction or metadata archive is missing. Stage I preserves its study-specific protocol; Stages II/III use the conditional nested CV described in the thesis. No new training is identified as a past published run.

## Input diagnostics and figure prerequisites

The dataset is expected inside the **local clone** under `Data/` or `data/`; an archive visible only on GitHub.com is not sufficient. Supported archive readers include `.zip`, `.csv.gz`, `.tar.gz`, `.tar`, `.7z` and `.rar`, with optional dependencies for 7z/RAR. Source-data diagnostic figures and ring relevance plots require the original Lorenzetti CSV; predictions stored in pickles alone do not contain all the 100-ring source observables. Results-specific figures also require the corresponding method's saved predictions and, for performance–cost plots, recorded numerical inference times and feature counts.

The output includes `paper_figure_coverage.csv` and the original chart metadata; the `Data/` archive remains unchanged and subsequent training uses new folders under `artifacts/new_runs/`.
