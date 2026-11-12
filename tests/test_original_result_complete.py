"""Regression tests for publication and original research result replay."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg',force=True)
from lorenzetti_repro.original_result_cells import ORIGINAL_RESULT_CELLS
from lorenzetti_repro.full_results_replay import generate_full_original_analyses
from lorenzetti_repro.original_replay import collect_shows
from lorenzetti_repro import original_plotting as op
from lorenzetti_repro.original_scenarios import STAGE23


def _frame(label,truth,fast,noise):
    alpha=truth/fast+noise
    return pd.DataFrame({'method-selection':[label]*len(truth),'mc_match_et':truth,
        'cluster_truthet':fast,'cluster_trutheta':np.linspace(-1,1,len(truth)),
        'ypred':alpha,'yte':truth/fast,'calib_et':fast*alpha})


def test_inline_source_result_cells_and_inventory():
    assert {'713','722','767'} <= ORIGINAL_RESULT_CELLS.keys()
    inventory=Path(__file__).parents[1]/'docs'/'ORIGINAL_RESULT_OUTPUT_INVENTORY.csv'
    found=pd.read_csv(inventory)
    assert {702,705,708,713,722,733,734,735,746,747,749,756,757,767,805,816,846,848,874,876} <= set(found.original_cell)


def test_global_original_comparison_panels_are_generated(tmp_path):
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex']=False
    rng=np.random.default_rng(123);n=110
    truth=np.linspace(12000,180000,n);fast=truth+rng.normal(0,1100,n)
    labels=['BDT no-Bin: PCA-10','BDT no-Bin: PCA-5','BDT no-Bin: ICA-5']
    frames={label:_frame(label,truth,fast,rng.normal(0,.0005+index*.0003,n))
            for index,label in enumerate(labels)}
    report=generate_full_original_analyses('tns',frames,{},tmp_path)
    assert {'comparison_metric_cell702','comparison_metric_all_cell705',
            'top_ten_residuals_cell749','rmse_iqr_quadrant_cell767'}.difference(report['paths']) == {'rmse_iqr_quadrant_cell767'}
    assert len(report['paths'])>=6
    assert all(p.is_file() and p.stat().st_size>1000 for group in report['paths'].values() for p in group)


def test_segmented_panel_origin_is_recorded(tmp_path):
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex']=False
    rng=np.random.default_rng(7)
    rows=[]
    for method,rep in [('GNN','Rings[1:100]'),('DNN','ICA-10')]:
        idx=STAGE23.index(rep)
        for energy,eta in [('All',(0,.43)),('All',(.43,.83)),('All',(.83,1.46)),
                           ((0,50000),'All'),((50000,100000),'All'),
                           ((100000,200000),'All'),((200000,300000),'All'),((300000,500000),'All')]:
            truth=rng.normal(1,.025,30);pred=truth+rng.normal(0,.01,30)
            rows.append(dict(method=f'{method} no-Bin',useRings=idx,
                et_range=energy,eta_range=eta,yte=truth,ypred=pred,IQRraw=.04,
                IQRpred=.02,MDraw=1,MDpred=1,e_mae=float(np.mean(abs(pred-truth))),
                e_rmse=float(np.sqrt(np.mean((pred-truth)**2))),
                e_mae_calib_n=float(np.mean(abs(1-1/pred))),
                e_mae_fast_n=float(np.mean(abs(1-1/truth)))))
    op.useRings_cenarios=STAGE23
    with collect_shows(tmp_path,'original_segmented') as figures:
        op.integrated_plots_in_ETA_ET_geral_with_ranges(pd.DataFrame(rows),STAGE23)
    source_lines={json.loads(p.with_suffix('.json').read_text())['source_line'] for p in figures}
    assert {552,698,976,1136} <= source_lines
    assert len(figures)>=10
