# Research repository requirements and verification checklist

## Requirements

1. Thesis and all three paper methods are normative for the scientific protocol. Do not silently substitute ATLAS-specific modifications found in the exploratory notebook.
2. One runnable notebook per publication, with model implementation, post-fit archive generation, archived replay, numerical analyses, and graphical functions.
3. Reuse author-source plotting functions from the exploratory notebook; preserve figure dimensions, fonts, labels, colors, legends, axes and call-specific filters. 
4. Published event archives are stored in `artifacts/pickles/`; new training cannot write there. New training writes `artifacts/new_runs/<article>/<unique-run-id>/` and must not overwrite an existing run.
5. Replay selection is explicit: `original`, `latest_new_run`, or `specific_new_run`.
6. Dataset archive remains in `data/` (or `Data/`); the original CSV was `Lorenzetti_electrons_merge_root.csv`; another historical processed name is `Lorenzetti_simulation_raw_20250815.csv`.
7. Use published definitions of D70/D30, conditional nested CV, model families, input features, RMSE, IQR, d0, D0, EAR and statistical procedures.
8. Source configurations must never be silently truncated to the first five/sorted top four in general plot-generation routines.
9. Optional legacy `*_times.pkl` containing models trained in `__main__` may be incompatible; their failures cannot abort separately archived event-level predictions. Do not invent latency data.
10. All user-facing repository docs, notebook Markdown and public instructions are in English. Preserve original scientific code/function identities where necessary.
11. Validate with a full `Run All` that produces actual plots/tables from representative *test* artifacts; full numerical identity requires access to the historic artifacts and dataset.
12. No false claims that new code was the precise historical executable if verification is not possible.

## Implementation checks (2026-11-06)

- [x] The three notebooks ran from the project root in original replay mode with correctly named synthetic test archives (IEEE TNS: 19 PNGs; Measurement: 19 PNGs; Information Fusion: 26 PNGs; no Python errors).
- [x] Each notebook inventories all event-level configurations in the selected archive.
- [x] A new training preserves every byte of the historical archive (sentinel tested).
- [x] Each new training creates a distinct directory; existing run folders are rejected.
- [x] Original and new replay sources are selected separately, including latest and explicit run IDs.
- [x] The optional timing loader catches incompatible model classes instead of terminating the prediction replay; direct historical GNN timing recovery is environment-dependent.
- [x] Segmented and integrated archives are now generated for newly completed training with the source notebook's functions and Lorenzetti bin limits, in addition to the six other original table types.
- [x] Output figures use the author's plotting functions, and inline views no longer reduce resolution to JPEG thumbnails.
- [x] All saved configurations enter at least one representation-group comparison or the full numerical ranking; no arbitrary first-five cutoff is applied to the general comparison loop.
- [x] The source dataset's exact original filename is documented for all articles.
- [ ] Published-numerical equality and pixel-level graph parity: cannot be certified without the original historical dataset, model archives, and execution environment.
- [ ] CPU-times for the historically serialized GNN: require that the original GNN classes and PyTorch Geometric be available, or a trusted numerical timing export.

The verification above does not imply that the entire historical training procedure was executed on the full dataset in this environment. It records completed checks and remaining external validation dependencies.
