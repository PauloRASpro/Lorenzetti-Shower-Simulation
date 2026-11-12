import numpy as np
import pandas as pd
import pytest
from lorenzetti_repro.legacy_artifacts import (
    catalog,load_method_bundle,select_event_frames,event_values,recognize_filename)
from lorenzetti_repro.artifact_plots import calculate_summary,render_replay

def _frame(n=65):
    et=np.linspace(5000,70000,n)
    ef=et*1.04
    alpha=et/ef*1.002
    return pd.DataFrame({'method':['GNN']*n,
        'method-selection':['GNN: Rings[1:100]']*n,
        'useRings':[0]*n,'yte':et/ef,'ypred':alpha,
        'mc_match_et':et,'cluster_truthet':ef,'cluster_trutheta':np.linspace(0,1.46,n),
        'calib_et':ef*alpha})

def test_inventory_and_guard(tmp_path):
    f=tmp_path/'GNN_noSEG_Full_DF.pkl.gz'
    _frame().to_pickle(f,compression='gzip',protocol=5)
    assert recognize_filename(f.name)['method']=='GNN'
    rows=catalog(tmp_path)
    assert len(rows)==1 and len(rows[0]['sha256'])==64
    with pytest.raises(PermissionError):load_method_bundle(tmp_path,'GNN')
    with pytest.raises(ValueError):load_method_bundle(tmp_path,'GNN',trust_pickle=True,
                                                      expected_sha256={f.name:'0'*64})
    result=load_method_bundle(tmp_path,'GNN',trust_pickle=True,
                              expected_sha256={f.name:rows[0]['sha256']})
    assert isinstance(result['GNN_noSEG_Full_DF'],pd.DataFrame)
    frames=select_event_frames({'GNN':result})
    assert list(frames)==['GNN: Rings[1:100]']
    vals=event_values(next(iter(frames.values())))
    assert np.allclose(vals.E_calib_MeV,vals.E_true_MeV*1.002)

def test_replay_visuals_from_trusted_synthetic_archive(tmp_path):
    frames={'GNN: Rings[1:100]':_frame()}
    summary=calculate_summary(frames)
    assert len(summary)==1 and summary.rmse_mev.iloc[0]>0
    out=tmp_path/'plots'
    figs=render_replay(frames,out,stage='synthetic')
    assert len(figs)>=7
    assert all((out/f).stat().st_size>1000 for f in figs)
