"""Figure-generating functions are traceable to experimental inputs, not PDFs."""
from pathlib import Path
import inspect
import nbformat
import pandas as pd
from lorenzetti_repro.publication_figures import FIGURES,generate
import lorenzetti_repro.publication_figures as pf

ROOT=Path(__file__).resolve().parents[1]

def test_figure_mapping_has_all_research_figures_but_no_copies():
    assert {i for group in FIGURES.values() for i in group} == set(range(19,46))
    assert not (ROOT/'figures'/'published').exists()
    assert 'pd.read_csv(Path(root)/REFERENCE' not in inspect.getsource(pf)

def test_missing_predictions_no_fake_figures(tmp_path):
    for paper in FIGURES:
        reg,missing=generate(paper,{}, {},tmp_path/paper)
        assert len(reg)==len(FIGURES[paper])
        assert all(row['status']=='missing_input' for row in reg)
        assert not list((tmp_path/paper).glob('*.png'))

def test_notebook_has_no_publication_image_embeds():
    for file in sorted((ROOT/'notebooks').glob('*.ipynb')):
        nb=nbformat.read(file,as_version=4)
        assert all('figures/published' not in cell.source for cell in nb.cells)
        assert all('![Figura' not in cell.source for cell in nb.cells)
        assert all(not cell.get('outputs') for cell in nb.cells if cell.cell_type=='code')
        assert any('TRUST_MY_PICKLES' in cell.source for cell in nb.cells if cell.cell_type=='code')
        assert any('EXECUTION_MODE' in cell.source for cell in nb.cells if cell.cell_type=='code')
