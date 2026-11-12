# Plotting functions and figure generation

The original experimental drawing functions are collected in `src/lorenzetti_repro/original_plotting.py` with their source-cell references. The source-cell plots for the neural gain and timing trade-off are in `original_inline_figures.py`. The article notebooks invoke these functions on archived results, preserving source layout parameters, legend positions, axes, markers and labels.

The experiment archives use `method-selection` labels and numeric `useRings` indices. These indices are interpreted using the stage-specific original scenarios in `original_scenarios.py`. Only the relevant configurations are supplied to the segmented plotting routines, as in the original post-loading experiment analysis.

All generated figure files are saved in their original resolution. Notebooks show lightweight inline previews to avoid overloading the Jupyter kernel. No static article figure is embedded as experimental output.
