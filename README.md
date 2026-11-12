# Machine Learning and Information Fusion for Electron Energy Calibration

Reproducible experimental workflows for event-by-event calibration of electron transverse energy using the Lorenzetti Showers benchmark. The calibration algorithms operate on the 100-ring calorimeter representation and predict the multiplicative correction applied to the Standard FastCalo energy estimate.

## Published studies

| Study | Executable notebook | Scope |
|---|---|---|
| IEEE Transactions on Nuclear Science (2026) | [`01_IEEE_TNS_2026.ipynb`](notebooks/01_IEEE_TNS_2026.ipynb) | Conventional regressors, boosted trees, feature selection/extraction and the historical Stage I procedure |
| Measurement (2026) | [`02_Measurement_2026.ipynb`](notebooks/02_Measurement_2026.ipynb) | Random Forest, gradient boosting, residual-correction hybrids and stacking |
| Information Fusion (2026 manuscript) | [`03_Information_Fusion_2026.ipynb`](notebooks/03_Information_Fusion_2026.ipynb) | Deep neural regression and regional relational–tabular GraphSAGE |

**Each notebook has two workflows:** (a) model development and training, including preservation of experimental outputs; (b) direct analysis of previously saved experiment DataFrames, including original plotting functions and statistical tables. Each paper notebook writes a figure-by-figure coverage index linked to experimental data, prediction artifacts, and analysis functions.

## Quick start

```bash
python -m pip install -e ".[notebooks]"
jupyter lab
```

Install compatible PyTorch and PyTorch Geometric packages for the graph model when required (see the package's `gnn` optional dependencies). Install a LaTeX distribution with `dvipng`, `lmodern` and `amsmath` to match the source notebook’s original TeX labels and font rendering. Open one of the notebooks above and use **Kernel → Restart & Run All**. The default `EXECUTION_MODE = "replay"` loads trusted historical outputs without retraining. Change it to `"train"` to execute new training runs. Each run is saved to a new `artifacts/new_runs/<article>/<UTC timestamp>_<ID>/` directory; no historical pickle is overwritten. To revisit a run, change `REPLAY_SOURCE` to `"latest_new_run"` or `"specific_new_run"`.

### Experimental input files

- **Events:** preserve the Lorenzetti archives in `data/` (or `Data/`). The original source notebook loads `Lorenzetti_electrons_merge_root.csv` and also identifies a processed file `Lorenzetti_simulation_raw_20250815.csv`. The notebooks decompress recognized archives and check the ring and energy columns. If no compatible archive is found, replay of stored predictions remains possible; new training requires the dataset.
- **Saved experiment DataFrames:** place the original `.pkl` or `.pkl.gz` files in `artifacts/pickles/` without renaming them. Examples: `BDT_noSEG_Full_DF.pkl.gz`, `integrated_result_RFBDT_noSEG_Full_DF.pkl.gz`, `GNN_noSEG_performance_metrics.pkl.gz`.
- **Prediction times:** the original `*_noSEG_times.pkl` files may contain large fitted Python estimators. The notebooks can extract numerical timings into `artifacts/metadata/prediction_times.csv` when trusted archives are available.
- **Replayed historical analyses:** figures and CSV summaries are saved under `results/replayed/<study>/paper_figures/`. **New training:** outputs are isolated in `artifacts/new_runs/<article>/<run-id>/` with `pickles/`, `trained_models/`, `manifests/`, and `figures/`. Eight types of archived DataFrames are written, including segmented and integrated results.

Pickle deserialization may execute code. Enable `TRUST_MY_PICKLES = True` **only for artifacts under your control**.

## Experimental protocol

The principal dataset contains 302,619 reconstructed events, split into D70 for development (211,833 events) and D30 for independent testing (90,786 events). The operator \(g_{ML}\) is fitted on D70. For Stages II and III, the conditional 5 × 5 nested cross-validation selects and internally evaluates \(f_{ML}\) before final fitting on D70 and evaluation on D30. Stage I retains its historical protocol. The Lorenzetti benchmark is ATLAS-inspired but is not the official ATLAS online reconstruction chain.

The original plotting functions are in `src/lorenzetti_repro/original_plotting.py`. Their source-cell references are recorded in `ORIGINAL_FUNCTION_MAP.json`. The `paper_figure_coverage.csv` written by each notebook reports every original data-driven paper figure and identifies missing inputs rather than replacing them with published numerical values or screenshots. See [the paper figure and source-cell index](docs/PAPER_FIGURE_INDEX.md) and [the execution guide](GETTING_STARTED.md).

## Tests

```bash
python -m pytest -q
```

## Citations

- P. R. A. da Silva *et al.*, “Machine Learning for Energy Calibration in High Energy Physics Experiments Using the Lorenzetti Showers Framework,” *IEEE Transactions on Nuclear Science* 73(5), 2026. DOI: [10.1109/TNS.2026.3686269](https://doi.org/10.1109/TNS.2026.3686269).
- P. R. A. da Silva *et al.*, “Hybrid tree-based machine learning for calorimetric energy calibration using the Lorenzetti Showers framework,” *Measurement* 290, 122771, 2026. DOI: [10.1016/j.measurement.2026.122771](https://doi.org/10.1016/j.measurement.2026.122771).
- P. R. A. da Silva and E. F. de Simas Filho, “Multi-Stage Neural Information Fusion for Fast Energy Calibration in Segmented Particle-Detector Calorimeters,” manuscript submitted in 2026.

See [`CITATION.cff`](CITATION.cff), [`references.bib`](references.bib) and [`papers/`](papers/).
