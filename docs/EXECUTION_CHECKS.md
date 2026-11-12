# Execution checks

For each study, the primary acceptance test is Jupyter **Restart & Run All** with the corresponding, trusted historical artifacts. The result must include recalculated metric tables, source-generated plots, and `publication_figure_coverage.csv`.

Before interpreting any comparison, verify that the input contains the same event order, truth energy, uncalibrated energy and prediction factor for each model. Check that the selected representation matches the publication and that segment boundaries come from the saved integrated result rather than being silently invented. Timing-based comparisons require archived numerical measurements, not current benchmark values.

The automated unit tests cover dataset schema validation, representations, metrics, conditional nested cross-validation and original plotting entry points. Full numeric equivalence with the publications must be evaluated separately against the original research artifacts.
