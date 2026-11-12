# Execution validation

The automated test suite checks dataset extraction, artifact integrity, model and representation interfaces, original plot function availability, result replay, historical feature-count parsing, run isolation, numerical metric definitions, and plot input ordering.

## Validated behavior

- Source files and article notebooks compile.
- ZIP and TAR source archives containing the expected Lorenzetti CSV are extracted without changing the compressed originals.
- Saved component names such as `[pc0, pc1, pc2]` are interpreted correctly when retrieving actual feature counts.
- Original residual and segmented plot routines accept correctly structured experimental arrays.
- Histogram and boxplot comparison data are passed in descending calibrated-energy RMSE order.
- The original `D0` ranking receives a Standard FastCalo reference computed from the same saved events and regressor-free metadata (zero model input attributes and prediction overhead).
- New-run isolation is checked by automated tests.

## Reproducibility boundaries

The numerical identity of every published figure cannot be established without executing the complete dataset and the historical saved predictions in a compatible environment. Testing with artificial event arrays validates the code paths and figure generation, but does not certify numerical or pixel-level equality with the journal results. Figures depending on trained-model timing metadata require those measured historical records. The distribution-fitting panels require the `distfit` Python package; exact historical font appearance additionally requires LaTeX and its original font packages.
