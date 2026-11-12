"""Regression coverage for the research notebook replay workflows."""
from __future__ import annotations
import io
import tarfile
import zipfile
import numpy as np
import pandas as pd
from lorenzetti_repro.dataset import find_dataset
from lorenzetti_repro.analysis_tables import archived_feature_counts
from scripts.unpack_lorenzetti import unpack
from lorenzetti_repro.original_replay import replay_original_figures
from lorenzetti_repro import original_plotting


def sample_csv(n=24):
    df=pd.DataFrame({str(i):np.linspace(1+i,3+i,n) for i in range(100)})
    df['mc_match_et']=np.linspace(15000,38000,n)
    df['cluster_truthet']=np.linspace(14900,37900,n)
    df['cluster_trutheta']=np.linspace(-1.2,1.2,n)
    return df.to_csv(index=False).encode()


def test_lorenzetti_archive_discovery_in_zip(tmp_path):
    folder=tmp_path/'Data';folder.mkdir()
    with zipfile.ZipFile(folder/'events.zip','w') as z:
        z.writestr('experiment/LORENZETTI_electrons_merge_root.csv',sample_csv())
    p=unpack(folder)
    assert p.name=='LORENZETTI_electrons_merge_root.csv'
    assert len(pd.read_csv(p))==24
    assert find_dataset(folder)==p


def test_lorenzetti_archive_discovery_in_tar(tmp_path):
    folder=tmp_path/'Data';folder.mkdir()
    raw=sample_csv()
    with tarfile.open(folder/'events.tar.gz','w:gz') as z:
        info=tarfile.TarInfo('somewhere/LORENZETTI_electrons_merge_root.csv')
        info.size=len(raw)
        z.addfile(info,io.BytesIO(raw))
    p=unpack(folder)
    assert len(pd.read_csv(p))==24


def test_unquoted_historical_selected_feature_names():
    df=pd.DataFrame({'Evaluation-Method':['BDT no-Bin']*3,
       'Feature-Selection-Method':['PCA-3','ICA-2','CORR-90'],
       'Selected-Features':['[pc0, pc1, pc2]','[ic0, ic1]','[1, 7, 21, 34]']})
    counts=archived_feature_counts({'BDT':{'BDT_noSEG_performance_metrics':df}})
    assert counts['BDT no-Bin: PCA-3']==3
    assert counts['BDT no-Bin: ICA-2']==2
    assert counts['BDT no-Bin: CORR-90']==4


def test_histogram_boxplot_receives_rmse_descending(tmp_path,monkeypatch):
    n=45
    truth=np.linspace(20000,40000,n)
    fast=truth+1000
    frames={}
    for name,noise in [('PCA-10',100),('PCA-5',250),('PCA-2',600)]:
        label='BDT no-Bin: '+name
        pred=(truth+noise)/fast
        frames[label]=pd.DataFrame({'method-selection':[label]*n,'mc_match_et':truth,
              'cluster_truthet':fast,'cluster_trutheta':np.linspace(-1,1,n),
              'ypred':pred,'yte':truth/fast,'calib_et':pred*fast})
    recorded=[]
    original=original_plotting.comparison_hist_boxplot_erros
    def intercept(data,labels,*a,**kw):
        recorded.append((labels,[float(np.sqrt(np.mean(np.asarray(e)**2))) for e in data]))
        # Preserve actual graphics in normal repository use; test only ordering.
    monkeypatch.setattr(original_plotting,'comparison_hist_boxplot_erros',intercept)
    replay_original_figures('tns',frames,{},tmp_path)
    candidates=[(lab,rmse) for lab,rmse in recorded if len(lab)>=3 and any('PCA-' in x for x in lab)]
    assert candidates
    for lab,rmse in candidates:
        idx=[i for i,k in enumerate(lab) if 'PCA-' in k]
        values=[rmse[i] for i in idx]
        assert values==sorted(values,reverse=True)


def test_d0_plot_contains_standard_fastcalo_reference(tmp_path):
    n=45
    truth=np.linspace(20000,40000,n)
    fast=truth+1200
    frames={}
    rows=[]
    for name,noise,features,t in [('PCA-10',150,10,.001),('PCA-5',300,5,.002)]:
        label='BDT no-Bin: '+name
        alpha=(truth+noise)/fast
        frames[label]=pd.DataFrame({'method-selection':[label]*n,'mc_match_et':truth,
           'cluster_truthet':fast,'cluster_trutheta':np.linspace(-1,1,n),
           'ypred':alpha,'yte':truth/fast,'calib_et':alpha*fast})
        rows.append({'configuration':label,'time_ms_per_event':t})
    root=tmp_path/'repo';out=root/'results'/'replayed'/'tns'/'paper_figures'/'trained_model_figures'
    metadata=root/'artifacts'/'metadata';metadata.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(metadata/'prediction_times.csv',index=False)
    perf=pd.DataFrame({'Evaluation-Method':['BDT no-Bin','BDT no-Bin'],
           'Feature-Selection-Method':['PCA-10','PCA-5'],
           'Selected-Features':['[pc0:pc9]','[pc0:pc4]']})
    bundles={'BDT':{'BDT_noSEG_performance_metrics':perf}}
    result=replay_original_figures('tns',frames,bundles,out)
    assert any(p.name.startswith('original_D0_ranking') for p in result['figure_paths']),result['notes']
    saved=pd.read_csv(out/'computed_D0_ranking.csv')
    assert 'Standard FastCalo:0' in saved['Method-Selection'].tolist()


def test_segmented_plot_has_original_required_statistics(tmp_path):
    from lorenzetti_repro.original_scenarios import STAGE23
    from lorenzetti_repro.original_plotting import integrated_plots_in_ETA_ET_geral_with_ranges
    from lorenzetti_repro.original_replay import collect_shows
    import matplotlib
    matplotlib.use('Agg',force=True)
    rng=np.random.default_rng(7)
    rows=[]
    for method,rep in [('GNN','Rings[1:100]'),('DNN','ICA-10')]:
        idx=STAGE23.index(rep)
        for energy,eta in [('All',(0,.43)),('All',(.43,.83)),('All',(.83,1.46)),
                           ((0,50000),'All'),((50000,100000),'All'),((100000,200000),'All'),
                           ((200000,300000),'All'),((300000,500000),'All')]:
            truth=rng.normal(1,.025,36);pred=truth+rng.normal(0,.01,36)
            rows.append(dict(method=f'{method} no-Bin',useRings=idx,et_range=energy,
                 eta_range=eta,yte=truth,ypred=pred,IQRraw=.04,IQRpred=.02,
                 MDraw=1,MDpred=1,e_mae=float(np.mean(np.abs(pred-truth))),
                 e_rmse=float(np.sqrt(np.mean((pred-truth)**2))),
                 e_mae_calib_n=float(np.mean(np.abs(1-1/pred))),
                 e_mae_fast_n=float(np.mean(np.abs(1-1/truth)))))
    with collect_shows(tmp_path,'original_segmented') as plots:
        integrated_plots_in_ETA_ET_geral_with_ranges(pd.DataFrame(rows),STAGE23)
    assert len(plots)>=10
    assert all(p.stat().st_size>1000 for p in plots)


def test_boxplot_model_order_descends_in_rmse(tmp_path,monkeypatch):
    import lorenzetti_repro.original_plotting as plot
    from lorenzetti_repro.original_scenarios import STAGE23
    plot.useRings_cenarios=STAGE23
    rows=[]
    for method,shift in [('GNN',.025),('DNN',.005)]:
        for energy,eta in [('All',(0,.5)),('All',(.5,1.4)),((0,50000),'All'),((50000,100000),'All')]:
            truth=np.ones(20);pred=truth+shift
            rows.append(dict(method=f'{method} no-Bin',useRings=0,
                et_range=energy,eta_range=eta,yte=truth,ypred=pred))
    captured=[]
    monkeypatch.setattr(plot,'box_plot_comparison_geral',lambda ranges,alphas,models,**kw: captured.append(models))
    plot.boxplot_comparison_improved(pd.DataFrame(rows))
    assert len(captured)==2
    assert captured[0][1:] == ['GNN no-Bin: Rings[1:100]','DNN no-Bin: Rings[1:100]']
