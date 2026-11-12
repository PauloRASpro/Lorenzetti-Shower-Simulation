"""Statistics and result tables calculated from saved event-level predictions.

Input columns are the archived notebook schema (mc_match_et, cluster_truthet,
ypred, optional cluster_trutheta). Numbers are derived from the supplied
predictions; no article image or published metric is used as input here.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
from .legacy_artifacts import event_values
from .metrics import shared_ear,delta_lambda_90
from .statistics import paired_residual_checks,directional_paired_inference

def configuration_feature_count(label: str) -> int:
    s=str(label).upper()
    if 'RINGS[1:100]' in s or 'RINGS100' in s or 'FULL' in s:return 100
    if 'RINGS/2' in s: return 50
    m=re.search(r'(PCA|ICA|CORR|MUTI|MI)-(\d+)(?!\d)',s)
    if m:
        kind,n=m.group(1),int(m.group(2))
        # Stage I labels CORR/MUTI-50/70/80/90 are RELEVANCE THRESHOLDS,
        # not cardinalities. Their dimensionality requires fitted D70 artifacts.
        if kind in ('CORR','MUTI','MI') and n in (50,70,80,90):return 0
        return n
    return 0  # Unknown must be resolved from archived metadata, not guessed.

def resolution_by_energy(e_true,e_estimate, n_bins=20):
    et=np.asarray(e_true,dtype=float)
    est=np.asarray(e_estimate,dtype=float)
    if np.any(et<=0):raise ValueError('Etrue must be positive for relative resolution')
    bins=np.linspace(np.min(et),np.max(et),n_bins+1)
    rows=[]
    for k,(low,high) in enumerate(zip(bins[:-1],bins[1:])):
        mask=(et>=low)&((et<high) if k<n_bins-1 else (et<=high))
        if mask.sum()<3:continue
        r=est[mask]/et[mask]
        rows.append(dict(Etrue_center_MeV=float((low+high)/2),N=int(sum(mask)),
                         relative_resolution=float(np.std(r,ddof=1)),
                         relative_resolution_se=float(np.std(r,ddof=1)/np.sqrt(2*(sum(mask)-1)))))
    return pd.DataFrame(rows)

def compute_event_tables(frames:dict):
    """Derive global errors, per-energy resolution, EAR90 and paired statistics."""
    performance=[];resolution=[];ear=[];stats=[]
    for label,raw in frames.items():
        v=event_values(raw)
        et=v.E_true_MeV.to_numpy();ef=v.E_fast_MeV.to_numpy();ec=v.E_calib_MeV.to_numpy()
        r=et-ec;b=et-ef
        def iqr(a):return float(np.percentile(a,75)-np.percentile(a,25))
        rmse=float(np.sqrt(np.mean(r*r)));brmse=float(np.sqrt(np.mean(b*b)))
        ii=iqr(r);bi=iqr(b)
        performance.append(dict(configuration=label,n=len(et),rmse_MeV=rmse,iqr_MeV=ii,
                                d0_MeV=float(np.hypot(rmse,ii)),F=configuration_feature_count(label),
                                rmse_reduction_pct=float(100*(1-rmse/brmse)),
                                iqr_reduction_pct=float(100*(1-ii/bi))))
        rr=resolution_by_energy(et,ec);rr['configuration']=label;resolution.append(rr)
        grid,cc,bb=shared_ear(et,ec,ef,points=1000)
        l90=delta_lambda_90(grid,cc);ref90=delta_lambda_90(grid,bb)
        ear.append(dict(configuration=label,delta_lambda90_MeV=l90,baseline_delta_lambda90_MeV=ref90,
                        delta_lambda90_reduction_pct=100*(1-l90/ref90) if ref90 and l90 is not None else np.nan))
        stats.append(dict(configuration=label,**directional_paired_inference(et,ef,v.alpha_calib.to_numpy())))
    return {'metrics':pd.DataFrame(performance),'resolution':pd.concat(resolution,ignore_index=True) if resolution else pd.DataFrame(),
            'ear90':pd.DataFrame(ear),'paired':pd.DataFrame(stats)}

def timing_from_archived_bundles(bundles):
    """Extract *numeric timing columns*; does not serialize trained estimators."""
    out=[]
    for method,bundle in bundles.items():
        key=f'{method}_noSEG_times'
        if key not in bundle:continue
        df=bundle[key]
        if not {'Method-Selection','Total Prediction-time (s)'}.issubset(df.columns):
            continue
        for _,r in df.iterrows():
            try: val=float(r['Total Prediction-time (s)'])*1000.
            except (ValueError,TypeError):continue
            if np.isfinite(val):
                out.append({'configuration':str(r['Method-Selection']), 'time_ms_per_event':val,
                            'F':configuration_feature_count(str(r['Method-Selection']))})
    return pd.DataFrame(out)

def compute_D0(performance:pd.DataFrame, timing:pd.DataFrame):
    """Eq. 5.12 D0: min-max normalize IQR,RMSE,F,T within this comparison only."""
    if performance.empty or timing.empty:raise ValueError('D0 requires per-event metrics AND archived per-event times')
    df=performance.merge(timing[['configuration','time_ms_per_event']],on='configuration',how='inner',validate='one_to_one')
    if df.empty:raise ValueError('No configuration names shared by metrics and times')
    if (df.F<=0).any():raise ValueError('Unknown external feature count F; cannot calculate D0')
    dimensions=['iqr_MeV','rmse_MeV','F','time_ms_per_event']
    x=df[dimensions].to_numpy(float)
    if not np.isfinite(x).all():raise ValueError('Non-finite D0 input')
    low=x.min(axis=0);diff=x.max(axis=0)-low
    normalized=np.divide(x-low,diff,out=np.zeros_like(x),where=diff>0)
    df['D0']=np.sqrt(np.sum(normalized**2,axis=1))
    return df.sort_values('D0').reset_index(drop=True)

def save_tables(tables,out):
    folder=Path(out);folder.mkdir(parents=True,exist_ok=True)
    for name,df in tables.items():
        if not df.empty:df.to_csv(folder/(name+'.csv'),index=False)
    return folder


def archived_feature_counts(bundles: dict) -> dict[str, int]:
    """Read representation dimensionality from original performance DataFrames.

    Historical ``Selected-Features`` entries contain either a nested column
    list or a compact notational range. CORR/MUTI threshold names are never
    treated as attribute counts. Ambiguous values are intentionally omitted.
    """
    import ast
    def size_of(x):
        if isinstance(x,(list,tuple,np.ndarray,pd.Index)):
            if len(x)==1:return size_of(x[0])
            return len(x) if all(not isinstance(t,(list,tuple,np.ndarray)) for t in x) else None
        if isinstance(x,str):
            stripped=x.strip()
            if not stripped:return None
            m=re.fullmatch(r'\[(?:pc|ic)(\d+):(?:pc|ic)(\d+)\]',stripped,flags=re.I)
            if m:return int(m.group(2))-int(m.group(1))+1
            try:
                parsed=ast.literal_eval(stripped)
            except (ValueError,SyntaxError):
                # The original notebook serializes component names without
                # Python quotes, e.g. [pc0, pc1, pc2] or [ic0, ic1].
                if stripped.startswith('[') and stripped.endswith(']'):
                    tokens=[t.strip() for t in stripped[1:-1].split(',') if t.strip()]
                    if tokens and all(re.fullmatch(r'(?:pc|ic)\d+|\d+',t,flags=re.I) for t in tokens):
                        return len(tokens)
                if stripped in ('Rings[1:100]','Rings[0:99]'):return 100
                if stripped=='Rings/2':return 50
                return None
            return size_of(parsed) if not isinstance(parsed,str) else None
        return None
    found={}
    for method,bundle in bundles.items():
        key=f'{method}_noSEG_performance_metrics'
        table=bundle.get(key)
        if not isinstance(table,pd.DataFrame) or table.empty:continue
        for _,row in table.iterrows():
            label=str(row.get('Evaluation-Method',f'{method} no-Bin')).strip()+': '+str(row.get('Feature-Selection-Method','')).strip()
            if label.endswith(':'):continue
            number=None
            for col in ('F','Number of Features','n_features'):
                if col in table.columns:
                    try:
                        v=int(row[col]);number=v if v>0 else None
                    except (ValueError,TypeError,OverflowError):pass
                    if number is not None:break
            if number is None and 'Selected-Features' in table.columns:
                number=size_of(row['Selected-Features'])
            if number is None:continue
            if label in found and found[label]!=number:
                found[label]=0 # conflicting historical records must be investigated
            else:found[label]=number
    return {label:number for label,number in found.items() if number>0}
