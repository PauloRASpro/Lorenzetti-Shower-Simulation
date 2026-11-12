"""Check the figure contract and the saved-artifact metadata interface."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from lorenzetti_repro.analysis_tables import archived_feature_counts
from lorenzetti_repro.original_article_cells import SOURCE_CELLS
from lorenzetti_repro.paper_figure_suite import PAPER_OUTPUTS
from lorenzetti_repro.original_figure_environment import ORIGINAL_LORENZETTI_LABEL,ORIGINAL_PREAMBLE

ROOT=Path(__file__).resolve().parents[1]

def test_article_results_figure_contract_is_complete_and_source_backed():
    assert {k:len(v) for k,v in PAPER_OUTPUTS.items()} == {
        'tns':12,'measurement':16,'information_fusion':16}
    assert set(PAPER_OUTPUTS['tns']) == set(range(4,16))
    assert set(PAPER_OUTPUTS['measurement']) == set(range(5,21))
    assert set(PAPER_OUTPUTS['information_fusion']) == set(range(5,21))
    assert all(groups for article in PAPER_OUTPUTS.values()
                  for _,groups in article.values())
    for name in ('736','739','741','60','64','68','72','74','75','76','118','120','163','164','199','206','207','208','209','211','212','213','214'):
        assert isinstance(SOURCE_CELLS[name],str) and len(SOURCE_CELLS[name].strip())>20

def test_original_latex_figure_style_is_preserved():
    assert r'\displaystyle \sqrt{s}' in ORIGINAL_LORENZETTI_LABEL
    assert 'lmodern' in ORIGINAL_PREAMBLE
    assert r'\usepackage{amsmath}' in ORIGINAL_PREAMBLE

def test_stage1_threshold_uses_archived_selected_feature_list_not_label():
    names=[str(x) for x in range(17)]
    sample=pd.DataFrame([
        {'Evaluation-Method':'BDT no-Bin','Feature-Selection-Method':'CORR-90','Selected-Features':[names]},
        {'Evaluation-Method':'BDT no-Bin','Feature-Selection-Method':'PCA-100','Selected-Features':['[pc0:pc99]']},
        {'Evaluation-Method':'BDT no-Bin','Feature-Selection-Method':'ICA-10','Selected-Features':[[str(i) for i in range(10)]]},
    ])
    result=archived_feature_counts({'BDT':{'BDT_noSEG_performance_metrics':sample}})
    assert result['BDT no-Bin: CORR-90']==17
    assert result['BDT no-Bin: PCA-100']==100
    assert result['BDT no-Bin: ICA-10']==10

def test_paper_notebooks_contain_no_embedded_pdf_graphs():
    import nbformat
    for file in (ROOT/'notebooks').glob('*.ipynb'):
        nb=nbformat.read(file,as_version=4)
        assert not any('image/png' in x.get('data',{}) for cell in nb.cells for x in cell.get('outputs',[]))
        code='\n'.join(''.join(c.source) for c in nb.cells if c.cell_type=='code')
        assert 'configure_original_figure_style' in code
        assert 'reproduce_paper_figure_set' in code
        assert "EXECUTION_MODE = 'replay'" in code
        assert 'make_new_run' in code
