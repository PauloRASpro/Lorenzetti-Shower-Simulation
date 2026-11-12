# Automated validation

Run `python -m pytest -q`. The suite covers calibration energy factors, numerical metrics, event-partition integrity, model wrappers, historical artifact schemas, scenario indexing, plotting-function availability and numerical consistency checks.

For the full notebook smoke test, supply trusted historical pickle files under `artifacts/pickles/` and use Jupyter Restart & Run All. The figure-coverage report lists all figures expected for the article. An empty output, missing original configuration or absent timing metadata is not treated as successful reproduction.
