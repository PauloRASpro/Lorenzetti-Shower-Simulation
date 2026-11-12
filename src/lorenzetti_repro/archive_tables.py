"""Non-training summaries of archived DataFrames recovered from trusted pickles."""
from pathlib import Path
import numpy as np
import pandas as pd

def _array_stats(obj):
    try:
        a=np.asarray(obj,dtype=float).reshape(-1)
        if not len(a) or not np.isfinite(a).all():return None
        return {'n':len(a),'rmse':float(np.sqrt(np.mean(a*a))),
                'iqr':float(np.quantile(a,.75)-np.quantile(a,.25))}
    except (TypeError,ValueError):
        return None

def export_stored_tables(bundles:dict[str,dict],output:str|Path)->list[str]:
    """Export existing stored tables WITHOUT recomputing or fitting models.

    The estimator in 'Best Model' is replaced by type name only. Large array
    cells in integrated results become counts to avoid accidental leakage/huge
    CSV exports. Original bundles remain unchanged in memory.
    """
    out=Path(output);out.mkdir(parents=True,exist_ok=True);files=[]
    for method,bundle in bundles.items():
        for stem,df in bundle.items():
            if stem.endswith('_noSEG_performance_metrics') or stem.endswith('_noSEG_cv_metrics_df') or stem.endswith('_noSEG_times'):
                tab=df.copy()
                if 'Best Model' in tab:
                    tab['Best Model']=tab['Best Model'].map(lambda x:f'{type(x).__module__}.{type(x).__qualname__}' if x is not None else '')
                dst=out/f'{stem}.csv';tab.to_csv(dst,index=False);files.append(dst.name)
            elif stem.startswith('integrated_result_') or stem.startswith('result_'):
                tab=df.copy()
                for col in ('yte','ypred'):
                    if col in tab:
                        tab[col+'_event_count']=tab[col].map(lambda a:len(a) if hasattr(a,'__len__') else 0)
                        tab=tab.drop(columns=col)
                dst=out/f'{stem}_binned_summary.csv';tab.to_csv(dst,index=False);files.append(dst.name)
            elif stem.endswith('_noSEG_erros_data') or stem.endswith('_noSEG_erros_data_alpha'):
                value_col='data_erros_alpha' if stem.endswith('_alpha') else 'data_erros'
                label_col='data_labels_alpha' if stem.endswith('_alpha') else 'data_labels'
                if value_col in df and label_col in df:
                    rows=[]
                    for _,row in df.iterrows():
                        stats=_array_stats(row[value_col]);
                        if stats is not None:rows.append({'configuration':row[label_col],**stats})
                    dst=out/f'{stem}_error_summaries.csv';pd.DataFrame(rows).to_csv(dst,index=False);files.append(dst.name)
    return files

def draw_original_binned_iqr(bundles:dict[str,dict],outdir:str|Path)->list[str]:
    """Plot the *stored* IQRraw/IQRpred arrays and original bin boundaries.

    This is closer to the historical plotting code than recomputing new bins.
    Does not claim exact original line styles/legend order.
    """
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    saved=[]
    for axis,col in [('eta','eta_range'),('energy','et_range')]:
        fig,ax=plt.subplots(figsize=(9,5));any_plotted=False;baseline=False
        for method,bundle in bundles.items():
            key=f'integrated_result_{method}_noSEG_Full_DF'
            if key not in bundle:continue
            df=bundle[key]
            if not {col,'IQRraw','IQRpred','method','useRings'}.issubset(df.columns):continue
            slice_df=df[df['et_range'].astype(str)=='All'] if axis=='eta' else df[df['eta_range'].astype(str)=='All']
            if slice_df.empty:continue
            for (m,ring),group in slice_df.groupby(['method','useRings']):
                xs=[];base=[];corrected=[]
                for _,r in group.iterrows():
                    span=r[col]
                    if not isinstance(span,(list,tuple,np.ndarray)) or len(span)!=2:continue
                    left,right=map(float,span)
                    mid=(left+right)/2/(1000 if axis=='energy' else 1)
                    xs.append(mid);base.append(float(r['IQRraw']));corrected.append(float(r['IQRpred']))
                if not xs:continue
                idx=np.argsort(xs)
                if not baseline:
                    ax.plot(np.asarray(xs)[idx],np.asarray(base)[idx],marker='o',label='Archived FastCalo reference');baseline=True
                ax.plot(np.asarray(xs)[idx],np.asarray(corrected)[idx],marker='.',label=f'{m} / useRings={ring}');any_plotted=True
        if any_plotted:
            ax.set(xlabel='E_true (GeV)' if axis=='energy' else '|η|',
                   ylabel='Stored IQR of α factors',
                   title='Historical integrated-result IQR (from original pickle)')
            ax.legend(fontsize=7);fig.tight_layout()
            path=out/f'archived_integrated_iqr_{axis}.png';fig.savefig(path,dpi=170);saved.append(path.name)
        plt.close(fig)
    return saved
