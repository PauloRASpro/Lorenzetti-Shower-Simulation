# Journal figure-to-code index

Each row identifies the paper figure, its analysis function, and the experimental inputs required for regeneration.

Original plotting statements and drawing functions are retained in `src/lorenzetti_repro/original_article_cells.py` and `src/lorenzetti_repro/original_plotting.py`. Their invocation is organized in `paper_figure_suite.py` and `original_replay.py`.

**Required data:** introductory diagnostics and feature relevance require the original Lorenzetti CSV. Regressor panels require the saved `*_noSEG_Full_DF.pkl` event-level results. Segmented panels require `integrated_result_*_noSEG_Full_DF.pkl`. Ranking/cost panels additionally require real numeric prediction times and the feature counts saved in historical performance tables.

If a group is absent, the resulting `paper_figure_coverage.csv` records `partial` or `missing_input` and the missing group. It never draws a surrogate figure with copied article values.

## Figure coverage

### IEEE Transactions on Nuclear Science

| Journal figure | Content | Original code / source required |
|---:|---|---|
| 4 | Energy distributions and true/fast scatter | original source notebook cell 120; original source notebook cell 118 |
| 5 | Standardized error and distribution goodness-of-fit | original source notebook cells 75–76; original source notebook cells 60, 64, 68, 72, 74 (distfit) |
| 6 | True calibration factor vs eta and ET | original source notebook cells 206–209; original source notebook cells 211–214 |
| 7 | 100 ring distributions | original source notebook cells 163–164 |
| 8 | CORR, MI, PCA and ICA feature map | original source notebook cell 199; D70 feature scores from source CSV |
| 9 | Calibrated energy residual distributions | comparison_hist_boxplot_erros, complete CORR family; comparison_hist_boxplot_erros, complete MUTI family; comparison_hist_boxplot_erros, complete PCA family; comparison_hist_boxplot_erros, complete ICA family |
| 10 | Top-ten accuracy/cost/complexity trade-off | original inline source cell 736 and ten actual ranked configurations; original inline source cell 739 and ten measured CPU time records; original inline source cell 741 and actual retained feature counts; comparison_metric_all plus measured time and feature metadata |
| 11 | IQR of calibration factor in eta and ET | integrated_plots_in_ETA_ET_geral_with_ranges, historical integrated bins |
| 12 | MAE and RMSE of calibration factor in eta and ET | integrated_plots_in_ETA_ET_geral_with_ranges, historical integrated bins |
| 13 | Calibration factor regional boxplots | boxplot_comparison_improved, historical integrated bins |
| 14 | Threshold acceptance curve | lambda_analysis, 1,000 grid points, event-level predictions |
| 15 | Error normal Q-Q comparison | qq_plot, original baseline residuals; qq_plot, selected calibrated residuals |

### Measurement

| Journal figure | Content | Original code / source required |
|---:|---|---|
| 5 | Energy distributions and true/fast scatter | original source notebook cell 120; original source notebook cell 118 |
| 6 | Standardized error and distribution goodness-of-fit | original source notebook cells 75–76; original source notebook cells 60, 64, 68, 72, 74 (distfit) |
| 7 | True calibration factor vs eta and ET | original source notebook cells 206–209; original source notebook cells 211–214 |
| 8 | 100 ring distributions | original source notebook cells 163–164 |
| 9 | CORR, MI, PCA and ICA feature map | original source notebook cell 199; D70 feature scores from source CSV |
| 10 | Calibration residual distributions by representation | comparison_hist_boxplot_erros, complete CORR family; comparison_hist_boxplot_erros, complete MUTI family; comparison_hist_boxplot_erros, complete PCA family; comparison_hist_boxplot_erros, complete ICA family |
| 11 | Joint error and complexity ranking | comparison_metric_all, saved event-level predictions; comparison_metric_all plus measured time and feature metadata |
| 12 | IQR in eta and energy regions | integrated_plots_in_ETA_ET_geral_with_ranges, historical integrated bins |
| 13 | MAE and RMSE in eta and energy regions | integrated_plots_in_ETA_ET_geral_with_ranges, historical integrated bins |
| 14 | Calibration-factor regional boxplots | boxplot_comparison_improved, historical integrated bins |
| 15 | Relative energy resolution curves | compute_resolution_by_energy and plot_resolution_curves |
| 16 | Threshold acceptance curves | lambda_analysis, 1,000 grid points, event-level predictions |
| 17 | Fast versus calibrated energy scatter | plota_energy_alpha_scatter_por_metodo, stored per-event predictions |
| 18 | Compact performance summary | original inline source cell 874 with actual paired predictions |
| 19 | Accuracy and efficiency trade-off | original inline source cell 876 and measured timing records |
| 20 | Error normal Q-Q comparison | qq_plot, original baseline residuals; qq_plot, selected calibrated residuals |

### Information Fusion

| Journal figure | Content | Original code / source required |
|---:|---|---|
| 5 | Energy distributions and true/fast scatter | original source notebook cell 120; original source notebook cell 118 |
| 6 | Standardized error and distribution goodness-of-fit | original source notebook cells 75–76; original source notebook cells 60, 64, 68, 72, 74 (distfit) |
| 7 | True calibration factor vs eta and ET | original source notebook cells 206–209; original source notebook cells 211–214 |
| 8 | 100 ring distributions | original source notebook cells 163–164 |
| 9 | CORR, MI, PCA and ICA feature map | original source notebook cell 199; D70 feature scores from source CSV |
| 10 | Calibration residual distributions by representation | comparison_hist_boxplot_erros, complete CORR family; comparison_hist_boxplot_erros, complete MUTI family; comparison_hist_boxplot_erros, complete PCA family; comparison_hist_boxplot_erros, complete ICA family |
| 11 | Joint error and complexity ranking | comparison_metric_all, saved event-level predictions; comparison_metric_all plus measured time and feature metadata |
| 12 | IQR in eta and energy regions | integrated_plots_in_ETA_ET_geral_with_ranges, historical integrated bins |
| 13 | MAE and RMSE in eta and energy regions | integrated_plots_in_ETA_ET_geral_with_ranges, historical integrated bins |
| 14 | Calibration-factor regional boxplots | boxplot_comparison_improved, historical integrated bins |
| 15 | Relative energy resolution curves | compute_resolution_by_energy and plot_resolution_curves |
| 16 | Threshold acceptance curves | lambda_analysis, 1,000 grid points, event-level predictions |
| 17 | Fast versus calibrated energy scatter | plota_energy_alpha_scatter_por_metodo, stored per-event predictions |
| 18 | Compact neural calibration performance | original inline source cell 874 and four matching neural predictors |
| 19 | Neural accuracy–efficiency trade-off | original inline source cell 876 and matching numeric timing records |
| 20 | Error normal Q-Q comparison | qq_plot, original baseline residuals; qq_plot, selected calibrated residuals |

## Original source notebook correspondence

- The original research notebook contains both data-generation/training cells and a later block that reloads saved DataFrames. The latter is the authoritative source for the display parameters and selection of archived configurations.
- A figure may contain several plotted panels. An entry is complete only when all its required source groups have emitted images; the generated coverage CSV is the execution-time record.
- Hardware-dependent timing and the fitted-estimator objects stored in `*_times.pkl` cannot be regenerated by reading per-event predictions alone. If numeric times are unavailable, the corresponding trade-off panel remains marked as requiring inputs.
- Non-experimental detector images, geometry diagrams, and externally prepared pile-up illustrations are excluded from the executable figure index: they have separate original sources rather than the research notebook plotting code.
- For archive security, load `.pkl` files only when they come from a trusted source.
