"""The Matplotlib and LaTeX settings used in the experimental source notebook.

Source: research notebook cells 3 and 8. This module changes only the display
backend settings, never the experimental calculation or fitted model.
"""
from __future__ import annotations
import shutil
import os
import matplotlib.pyplot as plt

ORIGINAL_PREAMBLE=(r'\usepackage[utf8]{inputenc}'
                   r'\usepackage[T1]{fontenc}'
                   r'\usepackage{lmodern}'
                   r'\usepackage{amsmath}')
ORIGINAL_LORENZETTI_LABEL=(
    r'\hspace{-.8cm} \emph{LORENZETTI Simulation} \\  '
    r'$\displaystyle \sqrt{s} = 13$ TeV (Single $e$)'
)


def configure_original_figure_style(*, require_latex=False):
    """Use the exact original rcParams if a LaTeX toolchain is installed.

    If the author opts out of mandatory TeX and the toolchain is unavailable,
    a MathText fallback is permitted but its typography is not identical.
    """
    available=bool(shutil.which('latex') and shutil.which('dvipng')) and os.environ.get('LORENZETTI_TEST_NO_LATEX','0')!='1'
    if require_latex and not available:
        raise RuntimeError(
            'The original publication figures use Matplotlib text.usetex=True. '
            'Install LaTeX, dvipng and the lmodern/amsmath packages to reproduce their typography.')
    plt.rcParams['text.usetex']=available
    if available:
        plt.rcParams['text.latex.preamble']=ORIGINAL_PREAMBLE
    return available


def source_figure_label():
    return ORIGINAL_LORENZETTI_LABEL if plt.rcParams.get('text.usetex',False) else 'LORENZETTI Simulation — 13 TeV (Single electron)'
