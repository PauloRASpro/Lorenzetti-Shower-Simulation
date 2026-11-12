"""Plots derived from archived event-level outputs; no training is performed.

These are scientifically equivalent diagnostic views, NOT pixel-identical
copies of the published figures unless explicitly validated against them.
All input energies and output residuals are in MeV.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from .legacy_artifacts import event_values

def _plt():
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    return plt

def _arrays(frames):
    return {name:event_values(frame) for name,frame in frames.items()}

def calculate_summary(frames):
    rows=[]
    for name,v in _arrays(frames).items():
        err=v.residual_calib_MeV.to_numpy()
        ef=v.residual_fast_MeV.to_numpy()
        q=lambda a:float(np.percentile(a,75)-np.percentile(a,25))
        rows.append({'configuration':name,'n_events':len(v),
            'rmse_mev':float(np.sqrt(np.mean(err**2))), 'iqr_mev':q(err),
            'd0_mev':float(np.hypot(np.sqrt(np.mean(err**2)),q(err))),
            'baseline_rmse_mev':float(np.sqrt(np.mean(ef**2))),
            'baseline_iqr_mev':q(ef),
            'rmse_reduction_pct':float(100*(1-np.sqrt(np.mean(err**2))/np.sqrt(np.mean(ef**2)))),
            'iqr_reduction_pct':float(100*(1-q(err)/q(ef))) if q(ef)>0 else np.nan})
    return pd.DataFrame(rows)

def draw_error_distributions(frames, out:Path, *,title:str=''):
    a=_arrays(frames);plt=_plt()
    fig, (ax1,ax2)=plt.subplots(2,1,figsize=(10,7),gridspec_kw={'height_ratios':[2,1]})
    first=next(iter(a.values()))
    fast=first.residual_fast_MeV.to_numpy()
    ax1.hist(fast,bins=65,histtype='step',density=True,label='Standard FastCalo')
    data=[fast];labels=['Standard FastCalo']
    for name,v in a.items():
        arr=v.residual_calib_MeV.to_numpy()
        ax1.hist(arr,bins=65,histtype='step',density=True,label=name)
        labels.append(name);data.append(arr)
    ax1.set(xlabel='E_true − E_estimated (MeV)',ylabel='Density',title=title or 'Calibration residuals')
    ax1.legend(fontsize=7)
    ax2.boxplot(data,vert=False,showfliers=False,tick_labels=labels)
    ax2.set_xlabel('Residual (MeV)');fig.tight_layout()
    fig.savefig(out,dpi=170,bbox_inches='tight');plt.close(fig)

def draw_resolution(frames,out:Path,n_bins:int=20):
    a=_arrays(frames);plt=_plt();fig,ax=plt.subplots(figsize=(9,5))
    for j,(name,v) in enumerate([('Standard FastCalo', next(iter(a.values()))),*a.items()]):
        truth=v.E_true_MeV.to_numpy()
        est=v.E_fast_MeV.to_numpy() if j==0 else v.E_calib_MeV.to_numpy()
        edges=np.unique(np.quantile(truth,np.linspace(0,1,n_bins+1)))
        xs=[];ys=[]
        for k,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            mask=(truth>=lo)&((truth<hi) if k<len(edges)-2 else (truth<=hi))
            if sum(mask)<30:continue
            response=est[mask]/truth[mask]
            xs.append(float(truth[mask].mean()/1000))
            ys.append(float(np.std(response,ddof=1)))
        ax.plot(xs,ys,marker='.',label=name)
    ax.set(xlabel='True transverse energy (GeV)',ylabel='σ(E_est / E_true)',title='Relative energy resolution by E_true bin')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out,dpi=170);plt.close(fig)

def _ear(a,b,n=1000):
    limit=6*max(float(np.std(a)),float(np.std(b)))
    delta=np.linspace(0,limit,n)
    # Strict inequality from the thesis: |epsilon| < delta
    def rates(x):
        ordered=np.sort(abs(np.asarray(x)))
        return np.searchsorted(ordered,delta,side='left')/len(ordered)
    return delta,rates(a),rates(b)

def draw_ear(frames,out:Path):
    a=_arrays(frames);plt=_plt();fig,ax=plt.subplots(figsize=(9,5))
    for name,v in a.items():
        fast=v.residual_fast_MeV.to_numpy();cal=v.residual_calib_MeV.to_numpy()
        d,fc,cc=_ear(fast,cal)
        ax.plot(d,fc,label=f'Standard FastCalo ({name})',linestyle='--',alpha=.65)
        ax.plot(d,cc,label=name)
    ax.axhline(.90,linestyle=':',color='gray')
    ax.set(xlabel='Δλ (MeV), truth-centered half-width',ylabel='EAR: fraction with |ε| < Δλ',
           title='Truth-referenced acceptance-window analysis (not trigger efficiency)')
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out,dpi=170);plt.close(fig)

def draw_scatter(frames,out:Path,max_points:int=25000):
    a=_arrays(frames);plt=_plt();fig,ax=plt.subplots(figsize=(7,7))
    for name,v in a.items():
        n=len(v);ind=np.arange(n) if n<=max_points else np.linspace(0,n-1,max_points,dtype=int)
        ax.scatter(v.E_true_MeV.to_numpy()[ind]/1000,v.E_calib_MeV.to_numpy()[ind]/1000,
                   s=2,alpha=.12,label=name)
    hi=max(v.E_true_MeV.max() for v in a.values())/1000
    ax.plot([0,hi],[0,hi],ls='--',color='gray',label='Ideal')
    ax.set(xlabel='E_true (GeV)',ylabel='E_calib (GeV)',title='Calibrated response vs MC truth')
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out,dpi=170);plt.close(fig)

def draw_qq(frames,out:Path):
    from scipy import stats
    a=_arrays(frames);plt=_plt();fig,ax=plt.subplots(figsize=(7,5))
    for name,v in a.items():
        x=np.sort(v.residual_calib_MeV.to_numpy())
        theo,emp=stats.probplot(x,dist='norm',fit=False)
        sel=np.linspace(0,len(x)-1,min(len(x),2000),dtype=int)
        ax.scatter(np.asarray(theo)[sel],np.asarray(emp)[sel],s=3,alpha=.4,label=name)
    ax.set(xlabel='Theoretical normal quantile',ylabel='Observed residual (MeV)',title='Q–Q diagnostics')
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out,dpi=170);plt.close(fig)

def draw_binned_alpha(frames,out:Path,axis:str='energy',metric:str='iqr'):
    a=_arrays(frames);plt=_plt();fig,ax=plt.subplots(figsize=(9,5))
    for name,v in a.items():
        raw=frames[name]
        if 'yte' not in raw:
            raise ValueError(f'{name}: yte missing; alpha_true needed')
        ideal=np.asarray(pd.to_numeric(raw['yte'],errors='raise'),dtype=float)
        pred=v.alpha_calib.to_numpy()
        if axis!='energy' and 'abs_eta' not in v:
            raise ValueError('abs_eta unavailable in saved frame')
        pos=v.E_true_MeV.to_numpy()/1000 if axis=='energy' else v.abs_eta.to_numpy()
        if axis!='energy' and 'abs_eta' not in v:
            raise ValueError('abs_eta unavailable in saved frame')
        edges=np.unique(np.quantile(pos,np.linspace(0,1,16)))
        xs=[];ys=[]
        for k,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            mask=(pos>=lo)&((pos<hi) if k<len(edges)-2 else (pos<=hi))
            if sum(mask)<15:continue
            vals=pred[mask] if metric=='iqr' else ideal[mask]-pred[mask]
            xs.append(float(pos[mask].mean()))
            if metric=='iqr':ys.append(float(np.percentile(vals,75)-np.percentile(vals,25)))
            else:ys.append(float(np.sqrt(np.mean(vals**2))))
        ax.plot(xs,ys,marker='.',label=name)
    ax.set(xlabel='E_true (GeV)' if axis=='energy' else '|η|',
           ylabel='IQR(α_calib)' if metric=='iqr' else 'RMSE(α_true − α_calib)',
           title=f'Factor {metric.upper()} by {axis} (regenerated diagnostic)')
    ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out,dpi=170);plt.close(fig)

def draw_reference_ranking(reference_csv:Path, out:Path):
    plt=_plt();df=pd.read_csv(reference_csv)
    if 'rmse_mev' in df: data=df[['configuration','rmse_mev']].dropna(); y='rmse_mev'; ylabel='RMSE (MeV)'
    elif 'rmse_gev' in df: data=df[['configuration','rmse_gev']].dropna(); y='rmse_gev'; ylabel='RMSE (GeV)'
    else:raise ValueError(f'Cannot plot ranking from {reference_csv}')
    data=data.sort_values(y,ascending=False)
    fig,ax=plt.subplots(figsize=(10,max(3,len(data)*.46)))
    ax.barh(data.configuration,data[y]);ax.set(xlabel=ylabel,title='Published reference values (not recalculated)')
    fig.tight_layout();fig.savefig(out,dpi=170,bbox_inches='tight');plt.close(fig)

def render_replay(frames:dict, outdir:str|Path, *, stage:str, reference_csv:str|Path|None=None) -> list[str]:
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    if not frames:raise ValueError('No archived per-event results to replay')
    outputs=[]
    for stem,fn in [('error_hist_box',draw_error_distributions),('resolution',draw_resolution),
                    ('ear',draw_ear),('energy_scatter',draw_scatter),('qq',draw_qq)]:
        dst=out/f'{stage}_{stem}.png';fn(frames,dst);outputs.append(dst.name)
    for axis in ('eta','energy'):
        if axis=='eta' and not all('cluster_trutheta' in f for f in frames.values()):continue
        for metric in ('iqr','rmse'):
            dst=out/f'{stage}_alpha_{metric}_{axis}.png'
            draw_binned_alpha(frames,dst,axis=axis,metric=metric);outputs.append(dst.name)
    if reference_csv:
        dst=out/f'{stage}_published_reference_ranking.png'
        draw_reference_ranking(Path(reference_csv),dst);outputs.append(dst.name)
    return outputs
