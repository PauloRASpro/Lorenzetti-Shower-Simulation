"""Data/artifact discovery and executable results for the three studies.

No trained model is fitted during historical-artifact replay. Figure values come
from DataFrames produced by the experimental notebook, never PDF screenshots or
hand-entered article metrics. Pickle reading is only appropriate for trusted files.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd

from .dataset import EXPECTED_EVENTS, find_dataset, read_dataset
from .legacy_artifacts import (ARTICLE_METHODS, load_method_bundle, catalog,
                               select_event_frames)
from .analysis_tables import compute_event_tables, save_tables
from .archive_tables import export_stored_tables, draw_original_binned_iqr
from .publication_figures import generate, FIGURES


def find_input_folders(root: Path):
    """Find source files in the repository without relying on the shell cwd."""
    root = Path(root).resolve()
    candidates = [root/'artifacts'/'pickles', root/'artifacts',root/'pickles',root/'Pickles',
                  root.parent/'artifacts'/'pickles',root.parent/'pickles']
    def nfiles(p):
        if not p.is_dir():return 0
        return sum(1 for f in p.rglob('*') if f.is_file() and (f.name.endswith('.pkl') or f.name.endswith('.pkl.gz') or f.name.endswith('.pickle')))
    candidates = sorted((p for p in candidates if nfiles(p)),key=nfiles,reverse=True)
    pickle_dir = candidates[0] if candidates else root/'artifacts'/'pickles'
    data_candidates = [root/'Data',root/'data',root.parent/'Data',root.parent/'data',root/'datasets',root.parent/'datasets']
    data_dir = next((p for p in data_candidates if p.is_dir() and any(f.name.lower().endswith(('.csv','.zip','.7z','.gz','.rar','.tar','.tgz')) for f in p.rglob('*') if f.is_file())),root/'data')
    return pickle_dir, data_dir


def prepare_data_optional(root: Path, *, extract:bool=False):
    """Return CSV path when available; auto-extract an archive only if requested."""
    _, data_dir=find_input_folders(root)
    try:return find_dataset(data_dir)
    except (ValueError,FileNotFoundError):
        if extract:
            from scripts.unpack_lorenzetti import unpack
            return unpack(data_dir)
    return None


def load_article_artifacts(article:str, root:Path, *, trust_local_pickle:bool=True, pickle_dir=None):
    """Recover the historical DataFrames by naming scheme from original cell 303.

    `trust_local_pickle` must be set only on files personally controlled by user.
    The trained models in huge *_times pickles are not required for replay.
    """
    root=Path(root)
    folder = Path(pickle_dir).resolve() if pickle_dir is not None else find_input_folders(root)[0]
    found=catalog(folder, checksum=False)
    present={r['method'] for r in found if r['known']}
    bundles={}
    if present and not trust_local_pickle:
        raise PermissionError('Historical pickle files can execute code. Only enable local pickle loading for your own trusted archive.')
    for method in ARTICLE_METHODS[article]:
        if method in present:
            d=load_method_bundle(folder,method,trust_pickle=True,include_model_objects=False)
            if d:bundles[method]=d
    # On some historical machines a full-event table was exported only as CSV.
    for method in ARTICLE_METHODS[article]:
        key=f'{method}_noSEG_Full_DF'
        if key in bundles.get(method,{}):continue
        hits=[]
        for base in [folder,root/'artifacts'/'csv',root/'data']:
            if base.exists():hits.extend(p for p in base.rglob(key+'.csv') if p.is_file())
        hits=list(dict.fromkeys(hits))
        if len(hits)==1:
            f=hits[0];bundles.setdefault(method,{})[key]=pd.read_csv(f)
    frames=select_event_frames(bundles)
    return folder,found,bundles,frames


def _vector(x):
    """Handle native arrays and CSV-exported list literals without eval()."""
    if isinstance(x,(np.ndarray,list,tuple,pd.Series)):
        try:return np.asarray(x,dtype=float).ravel()
        except (TypeError,ValueError):return np.array([],dtype=float)
    if isinstance(x,str):
        try:
            safe=re.sub(r'np\.(?:float64|float32)\(([-+0-9.eE]+)\)',r'\1',x)
            parsed=ast.literal_eval(safe)
            return np.asarray(parsed,dtype=float).ravel()
        except (SyntaxError,ValueError,TypeError):return np.array([],dtype=float)
    return np.array([],dtype=float)


def historic_method_tables(bundles:dict, out:Path):
    """Display/export original per-configuration CV, score and summary tables."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    all_rows={}; files=export_stored_tables(bundles,out)
    for method,bundle in bundles.items():
        for suffix in ('_noSEG_performance_metrics','_noSEG_cv_metrics_df'):
            name=f'{method}{suffix}'
            if name in bundle:
                d=bundle[name]
                if not d.empty:
                    all_rows[name]=d.copy()
    return all_rows,files


def alpha_only_results(bundles:dict, output:Path):
    """Analysis of α residuals where eventwise Etrue/Efast were not saved.

    These cannot substitute for physical-energy RMSE, E_true resolution, or
    paired inference without original event arrays, so no such metrics are
    manufactured here.
    """
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[]; vectors=[]
    for method,bundle in bundles.items():
        key=f'{method}_noSEG_erros_data_alpha'
        if key not in bundle:continue
        df=bundle[key]
        if not {'data_erros_alpha','data_labels_alpha'}.issubset(df.columns):continue
        for _,rec in df.iterrows():
            a=_vector(rec['data_erros_alpha'])
            if len(a)<2 or not np.isfinite(a).all():continue
            label=str(rec['data_labels_alpha'])
            rows.append({'configuration':label,'events':len(a),'alpha_rmse':float(np.sqrt(np.mean(a*a))),
                         'alpha_IQR':float(np.quantile(a,.75)-np.quantile(a,.25))})
            vectors.append((label,a))
    summary=pd.DataFrame(rows)
    files=[]
    if not summary.empty:
        summary.to_csv(output/'archived_alpha_residuals.csv',index=False)
        files.append('archived_alpha_residuals.csv')
        # Only a few labelled traces per plot to avoid rendering hundreds of curves.
        for k in range(0,len(vectors),8):
            fig,axs=plt.subplots(1,2,figsize=(12,4.4));part=vectors[k:k+8]
            for label,a in part:
                axs[0].hist(a,bins=65,density=True,histtype='step',linewidth=1.1,label=label)
            axs[0].legend(fontsize=6);axs[0].set(xlabel='α_true − α_pred',ylabel='Probability density')
            axs[1].boxplot([a for _,a in part],vert=False,showfliers=False)
            axs[1].set_yticks(np.arange(1,len(part)+1));axs[1].set_yticklabels([lab for lab,_ in part],fontsize=6)
            axs[1].set(xlabel='α residual')
            fig.tight_layout();name=f'alpha_error_panels_{k//8+1:02d}.png'
            fig.savefig(output/name,dpi=125);plt.close(fig);files.append(name)
    return summary,files


def historic_binned_results(bundles:dict,output:Path):
    """Generate numerical IQR and MAE/RMSE of α in original eta/Efast bins.

    Source: convert_noSEG_to_SEG_DF from original notebook cell 250.
    Historic bins are defined in E_fast, not assumed E_true.
    """
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    table=[]
    for method,bundle in bundles.items():
        for prefix in ('result_','integrated_result_'):
            key=f'{prefix}{method}_noSEG_Full_DF'
            if key not in bundle:continue
            df=bundle[key]
            if not {'yte','ypred','et_range','eta_range'}.issubset(df.columns):continue
            for _,rec in df.iterrows():
                aa=_vector(rec['yte']);pp=_vector(rec['ypred']);
                if len(aa)!=len(pp) or not len(aa):continue
                e=aa-pp
                table.append({'method':method,'dataset':prefix,'useRings':rec.get('useRings'),
                              'et_range':str(rec['et_range']),'eta_range':str(rec['eta_range']),
                              'n':len(e),'alpha_RMSE':float(np.sqrt(np.mean(e*e))),
                              'alpha_MAE':float(np.mean(abs(e))),
                              'alpha_IQR_pred':float(np.quantile(pp,.75)-np.quantile(pp,.25)),
                              'alpha_IQR_true':float(np.quantile(aa,.75)-np.quantile(aa,.25))})
    data=pd.DataFrame(table)
    files=[]
    if not data.empty:
        data.to_csv(output/'segmented_alpha_metrics.csv',index=False)
        files.append('segmented_alpha_metrics.csv')
        # Avoid replotting duplicates: published summary focuses on a few representative configurations.
        for name,axis in [('eta_range','eta'),('et_range','fast_energy')]:
            for method,g in data.groupby('method',sort=True):
                vals=g[(g.dataset=='result_') & (g[name]!='All')]
                if vals.empty:continue
                gg=vals.groupby(name,as_index=False).agg(alpha_RMSE=('alpha_RMSE','mean'),
                            alpha_IQR_pred=('alpha_IQR_pred','mean'),n=('n','sum'))
                if gg.empty:continue
                # Original intervals are provided as categorical labels; do not imply Etrue binning.
                fig,ax=plt.subplots(figsize=(9,4.1))
                pos=np.arange(len(gg));ax.plot(pos,gg.alpha_RMSE,'o-',label='α RMSE');ax.plot(pos,gg.alpha_IQR_pred,'.-',label='IQR(α_pred)')
                ax.set_xticks(pos);ax.set_xticklabels(gg[name].tolist(),rotation=45,ha='right',fontsize=7)
                ax.set(xlabel='Archived '+axis+' interval',ylabel='α metric');ax.legend();fig.tight_layout()
                filename=f'segmented_{method}_{axis}.png';fig.savefig(output/filename,dpi=125);plt.close(fig);files.append(filename)
    return data,files


def analyze_article(article:str,root:Path,*,data_csv:Path|None=None,trust_local_pickle:bool=True,pickle_dir=None):
    """One-call execution of historical analysis and all available figures."""
    root=Path(root).resolve()
    folder,inventory,bundles,frames=load_article_artifacts(article,root,trust_local_pickle=trust_local_pickle,pickle_dir=pickle_dir)
    if not bundles:
        raise FileNotFoundError(
            'No model results found. Copy the article pickles (e.g. BDT_noSEG_Full_DF.pkl.gz) '
            f'to {root/"artifacts"/"pickles"}. The notebook cannot derive published outputs from PDF tables.')
    dest=root/'results'/'replayed'/article;dest.mkdir(parents=True,exist_ok=True)
    tables,stored_files=historic_method_tables(bundles,dest/'tables_historical')
    alpha,alpha_files=alpha_only_results(bundles,dest/'alpha_results')
    segmented,segment_files=historic_binned_results(bundles,dest/'segmented_results')
    if data_csv is None:
        data_csv=prepare_data_optional(root,extract=False)
    figs,missing=generate(article,frames,bundles,dest/'publication_figures',data_csv=data_csv)
    result={
        'article':article,'pickle_folder':str(folder),'files_found':len(inventory),
        'methods_loaded':list(bundles),'full_event_configurations':list(frames),
        'archived_tables':list(tables),'alpha_error_configs':len(alpha),
        'binned_rows':len(segmented),'plots_generated':sum(f['status']=='generated' for f in figs),
        'plots_possible':len(figs),'publication_figure_results':figs,
        'missing_named_configurations':missing,
        'extra_generated_figures':alpha_files+segment_files,
        'output':str(dest),
    }
    (dest/'run_report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return result,{'tables':tables,'alpha':alpha,'binned':segmented,'figure_paths':[
        dest/'publication_figures'/r['file'] for r in figs if r['status']=='generated'],
        'extra_paths':[dest/'alpha_results'/f for f in alpha_files if f.endswith('.png')]+[
                       dest/'segmented_results'/f for f in segment_files if f.endswith('.png')]}
