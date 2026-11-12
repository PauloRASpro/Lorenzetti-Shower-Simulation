"""Render research figures through the author's original plotting functions.

The original function bodies (including fonts, layout, labels, axes and colors)
are stored in original_plotting.py; this module only supplies original event
arrays and archives, captures figures at plt.show(), and saves them. No paper
figure is read or inserted as an image.
"""
from __future__ import annotations
from pathlib import Path
from contextlib import contextmanager
import json
import inspect
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import original_plotting as op
from .legacy_artifacts import event_values
from .original_figure_environment import source_figure_label

@contextmanager
def collect_shows(target: Path, prefix: str):
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    original=plt.show
    captured=[]
    def save_show(*args,**kwargs):
        caller=inspect.currentframe().f_back
        source_line=caller.f_lineno if caller else None
        source_function=caller.f_code.co_name if caller else None
        for num in plt.get_fignums():
            fig=plt.figure(num)
            path=target/f'{prefix}_{len(captured)+1:02d}.png'
            fig.savefig(path,dpi=fig.dpi)
            # Preserve a machine-checkable record of the original plotting
            # configuration for review and figure-to-paper correspondence.
            chart_meta={'size_inches':[float(x) for x in fig.get_size_inches()],
                        'dpi':float(fig.dpi),'source_line':source_line,
                        'source_function':source_function,'axes':[]}
            for ax in fig.axes:
                legend=ax.get_legend()
                chart_meta['axes'].append({
                    'title':ax.get_title(), 'xlabel':ax.get_xlabel(),
                    'ylabel':ax.get_ylabel(),
                    'font_title':ax.title.get_fontsize(),
                    'font_xlabel':ax.xaxis.label.get_fontsize(),
                    'font_ylabel':ax.yaxis.label.get_fontsize(),
                    'legend_labels':([t.get_text() for t in legend.get_texts()]
                                     if legend is not None else []),
                    'legend_font_size':(legend.get_texts()[0].get_fontsize()
                                        if legend is not None and legend.get_texts() else None),
                    'legend_location':(str(legend._loc) if legend is not None else None),
                })
            path.with_suffix('.json').write_text(
                json.dumps(chart_meta,indent=2,ensure_ascii=False),encoding='utf8')
            captured.append(path)
            plt.close(fig)
    plt.show=save_show
    try: yield captured
    finally:
        plt.show=original
        plt.close('all')

def exact_configuration(label, model, representation):
    """Match complete historical method and representation names (PCA-10 != PCA-100)."""
    left,sep,right=str(label).partition(':')
    if not sep:return False
    model_name=left.strip().removesuffix(' no-Bin').strip()
    selection=right.strip().removesuffix('(B)').strip()
    return model_name.casefold()==model.casefold() and selection.casefold()==representation.casefold()

def _prepared(frames):
    out={}
    for label,frame in frames.items():
        d=event_values(frame)
        if len(d)>3:out[label]=d
    return out

def _same_event_order(rows):
    rows=list(rows)
    if not rows:return False
    b=rows[0]
    return all(len(a)==len(b) and np.allclose(a.E_true_MeV,b.E_true_MeV,rtol=1e-12,atol=1e-6)
               and np.allclose(a.E_fast_MeV,b.E_fast_MeV,rtol=1e-12,atol=1e-6) for a in rows[1:])

def replay_original_figures(article: str, frames:dict, bundles:dict, output:Path, data_csv=None, *, allow_nonpublication_reference=False):
    """Use exact original plotting routines, not replacement plot templates.

    Missing/invalid historic data is reported per analysis; never replaced with
    article metrics, invented feature counts or example simulated curves.
    """
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    data=_prepared(frames); results=[]; notes=[]
    op.full_data_text_label=source_figure_label()
    if not data:
        return {'figure_paths':[], 'notes':['No event-level prediction frames loaded']}
    labels=list(data)
    def rmse_for(label):
        residual=np.asarray(data[label].residual_calib_MeV,dtype=float)
        return float(np.sqrt(np.mean(residual**2)))
    # Historical boxplot ordering: larger RMSE to smaller RMSE. The original
    # drawing functions receive ordered vectors; do not modify their formatting.
    by_family={family:sorted([k for k in labels if re.search(r':\s*'+family+r'[-_]',k,re.I)],
                       key=rmse_for,reverse=True)
               for family in ('CORR','MUTI','PCA','ICA')}
    sorted_names=sorted(labels,key=rmse_for,reverse=True)
    # Select the representative configuration actually reported for each article,
    # not the smallest error among whatever unrelated saved files happen to exist.
    article_leaders={
        'tns':('BDT','PCA-10'),
        'measurement':('RFBDT','Rings[1:100]'),
        'information_fusion':('GNN','Rings[1:100]'),
    }
    model,selection=article_leaders[article]
    representatives=[k for k in labels if exact_configuration(k,model,selection)]
    if not representatives:
        if not allow_nonpublication_reference:
            notes.append({'analysis':'article_reference','reason':
                 f'Expected {model} no-Bin: {selection} not found; other available configurations are not substituted.'})
            return {'figure_paths':[], 'notes':notes, 'methods':len(data)}
        best=sorted_names[0]
        notes.append({'analysis':'new_run_reference','reason':
              f'Published reference {model}: {selection} absent from this new run. '
              f'Illustrating available newly trained configuration {best}; not a historical publication result.'})
    else:
        best=representatives[0]
    first=data[best]
    base=first.residual_fast_MeV.to_numpy(float)
    def attempt(tag,fn):
        try:
            with collect_shows(out,tag) as saved: fn()
            if not saved:raise RuntimeError('Original routine returned without a visible figure')
            results.extend(saved)
        except Exception as exc:
            notes.append({'analysis':tag,'status':'not generated','reason':f'{type(exc).__name__}: {exc}'})
    # The experimental notebook's comparison (cells 746–747) plots every
    # archived configuration in each feature family on one canvas. Splitting
    # into four-model subsets alters both the original composition and layout.
    for family,selected in by_family.items():
        if not selected: continue
        chosen=[first.residual_fast_MeV.to_numpy(float)]
        labels_full=['Standard FastCalo']
        for key in selected:
            chosen.append(data[key].residual_calib_MeV.to_numpy(float))
            labels_full.append(key)
        attempt(f'original_hist_boxplot_{family.lower()}',
            lambda a=chosen,b=labels_full:
                op.comparison_hist_boxplot_erros(a,b,op.full_data_text_label,
                    xtextlabel=r'$E_{true} - E_{calib}$ (MeV)'))
    # The original d0 ranking (cell 88). Preserve chart aesthetics and labels.
    table=[]
    for key,d in data.items():
        e=d.residual_calib_MeV.to_numpy(float)
        table.append({'data_labels':key,'d0':float(np.hypot(np.sqrt(np.mean(e**2)),np.subtract(*np.quantile(e,[.75,.25]))))})
    table.append({'data_labels':'Standard FastCalo','d0':float(np.hypot(np.sqrt(np.mean(base**2)),np.subtract(*np.quantile(base,[.75,.25]))))})
    d0=pd.DataFrame(table).sort_values('d0').reset_index(drop=True)
    attempt('original_d0_ranking',lambda:op.comparison_metric_all(d0,op.full_data_text_label,rank='d0',show='d0'))
    # Original resolution routines (cells 782–783), original graph typography.
    Et=first.E_true_MeV.to_numpy(float)
    Ef=first.E_fast_MeV.to_numpy(float)
    Ec=first.E_calib_MeV.to_numpy(float)
    attempt('original_resolution',lambda:op.plotar_resolucao(Et,Ef,Ec,
                                'Standard FastCalo Energy',best))
    # Each original multi-resolution plot displays a small comparison set.
    # Produce additional panels rather than silently discarding other archived models.
    for start in range(0,len(sorted_names),4):
        names=sorted_names[start:start+4]
        if not _same_event_order([first]+[data[k] for k in names]):
            notes.append({'analysis':f'resolution_comparison_{start//4+1:02d}',
                          'reason':'Event order differs; cross-model overlay omitted'})
            continue
        attempt(f'original_resolution_comparison_{start//4+1:02d}',
            lambda names=names: op.plotar_resolucao_multipla(
                Et,Ef,[data[k].E_calib_MeV.to_numpy(float) for k in names],
                'Standard FastCalo Energy',names))
    attempt('original_scatter',lambda:op.plota_energy_alpha_scatter_por_metodo(
            frames[best],op.full_data_text_label))
    attempt('original_qq_fast',lambda:op.qq_plot(base,'Standard FastCalo'))
    attempt('original_qq_calibrated',lambda:op.qq_plot(first.residual_calib_MeV.to_numpy(float),best))
    # lambda_analysis is the original O(1000*N) analysis; keep it unmodified.
    attempt('original_EAR',lambda:op.lambda_analysis(base,first.residual_calib_MeV.to_numpy(float),
                   'Standard FastCalo',best,op.full_data_text_label))
    # The publications show more than one representative configuration.
    # Render each selected example with the same original (unmodified) figure
    # functions rather than silently replacing its style with a generic chart.
    selections={'tns':[('BDT','PCA-10')],
                'measurement':[('RFBDT','Rings[1:100]'),('BDT','ICA-5')],
                'information_fusion':[('GNN','Rings[1:100]'),('DNN','ICA-10'),
                                      ('DNN','CORR-10'),('DNN','Rings[1:100]')]}
    for mod,rep in selections[article]:
        matches=[k for k in data if exact_configuration(k,mod,rep)]
        if len(matches)!=1:
            notes.append({'analysis':f'{mod}_{rep}', 'reason':'Original prediction configuration missing or ambiguous'})
            continue
        name=matches[0]
        if name==best:continue
        v=data[name]
        tag=re.sub(r'[^A-Za-z0-9_-]','_',name)
        attempt(f'original_resolution_{tag}',lambda d=v,k=name:op.plotar_resolucao(
            d.E_true_MeV.to_numpy(float), d.E_fast_MeV.to_numpy(float),
            d.E_calib_MeV.to_numpy(float),'Standard FastCalo Energy',k))
        attempt(f'original_scatter_{tag}',
            lambda k=name:op.plota_energy_alpha_scatter_por_metodo(frames[k],op.full_data_text_label))
        attempt(f'original_qq_{tag}',lambda d=v,k=name:op.qq_plot(
            d.residual_calib_MeV.to_numpy(float),k))
        attempt(f'original_EAR_{tag}',lambda d=v,k=name:op.lambda_analysis(
            d.residual_fast_MeV.to_numpy(float),d.residual_calib_MeV.to_numpy(float),
            'Standard FastCalo',k,op.full_data_text_label))

    # Article III Figure 37 uses exactly original notebook cell 874 and four
    # independently computed performance comparisons (not manuscript values).
    if article=='information_fusion':
        from .original_inline_figures import plot_original_fig37
        expected=[('DNN','CORR-10'),('DNN','ICA-10'),('DNN','Rings[1:100]'),('GNN','Rings[1:100]')]
        found=[]
        for mod,rep in expected:
            hits=[lab for lab in data if exact_configuration(lab,mod,rep)]
            if len(hits)!=1:
                notes.append({'analysis':'original_Fig37',
                              'reason':f'Expected exactly one {mod}: {rep}; found {len(hits)}'})
                found=[];break
            found.append(hits[0])
        if found:
            from . import original_plotting as _orig
            def calc_ear90(e_calib,e_fast):
                grid=np.linspace(0,6*max(np.std(e_calib),np.std(e_fast)),1000)
                acc=np.searchsorted(np.sort(np.abs(e_calib)),grid,side='left')/len(e_calib)
                idx=np.flatnonzero(acc>=.90)
                return float(grid[idx[0]]) if len(idx) else np.nan
            if _same_event_order([data[k] for k in found]):
                et=first.E_true_MeV.to_numpy(float)
                fastest=first.E_fast_MeV.to_numpy(float)
                ef=et-fastest
                rmse_fast=float(np.sqrt(np.mean(ef**2)))
                iqr_fast=float(np.diff(np.quantile(ef,[.25,.75]))[0])
                res_fast=_orig.global_resolution_with_uncertainty(fastest,et)[0]
                ear_fast=calc_ear90(ef,ef)
                # Each original EAR chart uses a shared scan maximum. Compute
                # threshold pairs on a common grid for mathematically fair gain.
                def ear_pair(e):
                    top=6*max(np.std(e),np.std(ef))
                    grid=np.linspace(0,top,1000)
                    def q(a):
                        acc=np.searchsorted(np.sort(abs(a)),grid,side='left')/len(a)
                        pos=np.flatnonzero(acc>=.9)
                        return float(grid[pos[0]]) if len(pos) else np.nan
                    return q(ef),q(e)
                rows=[]
                for k in found:
                    d=data[k];e=d.residual_calib_MeV.to_numpy(float)
                    R=_orig.global_resolution_with_uncertainty(d.E_calib_MeV.to_numpy(float),et)[0]
                    fb,cb=ear_pair(e)
                    rows.append([100*(1-np.sqrt(np.mean(e**2))/rmse_fast),
                                 100*(1-float(np.diff(np.quantile(e,[.25,.75]))[0])/iqr_fast),
                                 100*(1-R/res_fast), 100*(1-cb/fb)])
                attempt('original_Fig37_gain_summary',lambda:plot_original_fig37(rows,['Standard FastCalo']+found))
                # Original Fig. 38 code is kept as cell 876; it requires measured,
                # per-configuration CPU timings, not fabricated timing constants.
                csv=out.parents[3]/'artifacts'/'metadata'/'prediction_times.csv' if len(out.parents)>3 else None
                # Also accept an explicit metadata CSV beside the repository artifacts.
                if csv is None or not csv.is_file():
                    possible=[p for p in out.parents if (p/'artifacts'/'metadata'/'prediction_times.csv').is_file()]
                    csv=(possible[0]/'artifacts'/'metadata'/'prediction_times.csv') if possible else None
                if csv is not None and csv.is_file():
                    timing=pd.read_csv(csv)
                    if {'configuration','time_ms_per_event'}.issubset(timing.columns):
                        wanted=[timing.loc[timing.configuration==k,'time_ms_per_event'] for k in found]
                        if all(len(v)==1 and pd.notna(v.iloc[0]) and float(v.iloc[0])>0 for v in wanted):
                            from .original_inline_figures import plot_original_fig38
                            d0ref=float(np.hypot(rmse_fast,iqr_fast))
                            d0s=[d0ref]+[float(np.hypot(np.sqrt(np.mean(data[k].residual_calib_MeV**2)),
                                np.diff(np.quantile(data[k].residual_calib_MeV,[.25,.75]))[0])) for k in found]
                            attempt('original_Fig38_tradeoff',lambda:plot_original_fig38(
                                ['Standard FastCalo']+found,[0]+[float(v.iloc[0]) for v in wanted],
                                d0s,[0,10,10,100,100]))
                        else:notes.append({'analysis':'original_Fig38','reason':'Missing measured CPU times for all four configurations'})
                    else:notes.append({'analysis':'original_Fig38','reason':'Timing CSV requires configuration,time_ms_per_event'})
                else:notes.append({'analysis':'original_Fig38','reason':'Generate artifacts/metadata/prediction_times.csv from original *_times.pkl files'})
            else:
                notes.append({'analysis':'original_Fig37',
                              'reason':'Historic event ordering differs among the four configurations'})

    # The plotting functions expect the same complete ETA/ET grid for each
    # configuration. The experimental notebook explicitly selected its best
    # configurations before calling the plotting functions (cells 843-848).
    # Passing every stored configuration can exceed its marker list and can
    # produce incomplete grids. Restrict the call to the published configurations.
    from .original_scenarios import STAGE1, STAGE23
    scenarios=STAGE1 if article=='tns' else STAGE23
    article_options={
      'tns':[('BDT','PCA-10'),('BDT','CORR-90'),('BDT','MUTI-90'),('BDT','ICA-10')],
      'measurement':[('RFBDT','Rings[1:100]'),('BDT','ICA-5'),('BDT','Rings[1:100]'),('RFOR','Rings[1:100]')],
      'information_fusion':[('GNN','Rings[1:100]'),('DNN','Rings[1:100]'),('DNN','ICA-10'),('DNN','CORR-10')],
    }
    integrated=[]
    for method,bundle in bundles.items():
        key=f'integrated_result_{method}_noSEG_Full_DF'
        if key not in bundle or bundle[key].empty:continue
        d=bundle[key].copy()
        # The original source cell 250 derives these six quantities from the
        # archived per-bin alpha vectors. Restore only columns missing from a
        # legacy export; never replace a recorded value.
        expected_metrics={'IQRraw','IQRpred','MDraw','MDpred',
                          'e_mae','e_rmse','e_mae_calib_n','e_mae_fast_n'}
        missing_metrics=expected_metrics.difference(d.columns)
        if missing_metrics and {'yte','ypred'}.issubset(d.columns):
            import ast
            def to_vector(value):
                if isinstance(value,str):
                    try:value=ast.literal_eval(value)
                    except (ValueError,SyntaxError):return np.array([],float)
                try:return np.asarray(value,dtype=float).reshape(-1)
                except (ValueError,TypeError):return np.array([],float)
            for col in missing_metrics:d[col]=np.nan
            for row_id,record in d.iterrows():
                yte=to_vector(record['yte']);ypred=to_vector(record['ypred'])
                if not len(yte) or len(yte)!=len(ypred) or np.any(yte==0) or np.any(ypred==0):
                    continue
                from scipy.stats import iqr as _iqr
                formula={
                  'IQRraw':float(_iqr(yte)),
                  'IQRpred':float(_iqr(yte/ypred)),
                  'MDraw':float(np.median(yte)),
                  'MDpred':float(np.median(yte/ypred)),
                  'e_mae':float(np.mean(np.abs(ypred-yte))),
                  'e_rmse':float(np.sqrt(np.mean((ypred-yte)**2))),
                  'e_mae_calib_n':float(np.mean(np.abs(1-1/ypred))),
                  'e_mae_fast_n':float(np.mean(np.abs(1-1/yte))),
                }
                for col in missing_metrics:d.at[row_id,col]=formula[col]
        required={'method','useRings','et_range','eta_range','IQRraw','IQRpred','MDraw','MDpred',
                  'e_mae','e_rmse','e_mae_calib_n','e_mae_fast_n','yte','ypred'}
        if not required.issubset(d.columns):
            notes.append({'analysis':'segmented_input', 'reason':f'{key}: missing {sorted(required-set(d.columns))}'})
            continue
        def original_scenario_index(v):
            if isinstance(v,(int,np.integer)):return int(v)
            if isinstance(v,str) and v.strip().isdigit():return int(v)
            matches=[i for i,rep in enumerate(scenarios) if str(rep)==str(v)]
            if len(matches)!=1:raise ValueError(f'Unknown representation {v!r}')
            return matches[0]
        try:d['useRings']=d.useRings.map(original_scenario_index)
        except (ValueError,TypeError) as exc:
            notes.append({'analysis':'segmented_input','reason':f'{key}: {exc}'});continue
        integrated.append(d)
    if integrated:
        union=pd.concat(integrated,ignore_index=True)
        selected=[]
        for mod,rep in article_options[article]:
            if rep not in scenarios:
                notes.append({'analysis':'segmented_input','reason':f'{rep} is outside the historical scenarios'});continue
            idx=scenarios.index(rep)
            rows=union[(union.method.astype(str).str.strip()==f'{mod} no-Bin') & (union.useRings==idx)].copy()
            if rows.empty:
                notes.append({'analysis':'segmented_input','reason':f'No integrated bins for {mod}: {rep}'});continue
            eta=rows[rows.et_range=='All']
            energy=rows[rows.eta_range=='All']
            if eta.empty or energy.empty:
                notes.append({'analysis':'segmented_input','reason':f'Incomplete ETA/ET integration for {mod}: {rep}'});continue
            if eta.eta_range.duplicated().any() or energy.et_range.duplicated().any():
                notes.append({'analysis':'segmented_input','reason':f'Duplicate ETA/ET intervals for {mod}: {rep}'});continue
            selected.append(rows)
        if selected:
            # Original calls use all selected curves; keep exactly their
            # plotting function and labels while presenting consistent bins.
            import ast
            def canon(v):
                if isinstance(v,str) and v!='All':
                    try:return tuple(ast.literal_eval(v))
                    except (SyntaxError,ValueError):return v
                if isinstance(v,np.ndarray):return tuple(v.tolist())
                if isinstance(v,list):return tuple(v)
                return v
            for d in selected:
                d['eta_range']=d.eta_range.map(canon)
                d['et_range']=d.et_range.map(canon)
            common_eta=set.intersection(*(set(d.loc[d.et_range=='All','eta_range']) for d in selected))
            common_et=set.intersection(*(set(d.loc[d.eta_range=='All','et_range']) for d in selected))
            if common_eta and common_et:
                tab=pd.concat([d[((d.et_range=='All') & d.eta_range.isin(common_eta)) |
                                 ((d.eta_range=='All') & d.et_range.isin(common_et))] for d in selected],ignore_index=True)
                op.useRings_cenarios=scenarios
                # Original cell 846 + cell 848, preserving original rendering code.
                attempt('original_segmented',lambda:op.integrated_plots_in_ETA_ET_geral_with_ranges(
                    tab,scenarios,graph_prefix=article))
                attempt('original_segmented_boxplot',lambda:op.boxplot_comparison_improved(
                    tab,graph_prefix=article))
            else:notes.append({'analysis':'original_segmented','reason':'No common historical ETA and ET bins across article configurations'})
        else:notes.append({'analysis':'original_segmented','reason':'No historical selected configuration with integrated ETA/ET bins'})
    else:notes.append({'analysis':'original_segmented','reason':'No archived integrated_result_* DataFrames'})
    # Additional analyses actually called after the artifact-loading cell in
    # the experimental notebook: cells 705–708, 733–757, 782–783 and 814–816.
    # Continue using the original figure-generating functions and layout.
    shown=sorted(labels,key=rmse_for,reverse=True)[:min(len(labels),10)]
    selected_data=[data[k] for k in shown]
    selected_residuals=[d.residual_calib_MeV.to_numpy(float) for d in selected_data]
    if selected_residuals:
        attempt('original_top10_residual_hist_box', lambda: op.comparison_hist_boxplot_erros(
            [base]+selected_residuals,['Standard FastCalo']+shown,op.full_data_text_label,
            xtextlabel=r'$E_{true}-E_{calib}$ (MeV)'))
        # Original cell 82 + cell 3136: box plot with the author's palette.
        attempt('original_top10_boxplot', lambda: op.comparison_boxplot(
            [base]+selected_residuals,['Standard FastCalo']+shown,op.full_data_text_label,
            xtextlabel=r'$E_{true}-E_{calib}$ (MeV)'))
    if _same_event_order(selected_data[:min(4,len(selected_data))]):
        subset=shown[:4]
        methods={'Standard FastCalo':first.E_fast_MeV.to_numpy(float)}
        methods.update({k:data[k].E_calib_MeV.to_numpy(float) for k in subset})
        def original_resolution_with_errorbars():
            curves=op.compute_resolution_by_energy(
                Et_true=first.E_true_MeV.to_numpy(float), methods=methods,
                n_bins=20, bin_strategy='width', min_count=100, robust=False,
                use_full_SE_for_R=True, bootstrap_B=None)
            op.plot_resolution_curves(curves,methods_order=list(methods),
                                     show_errorbars=True,use_bootstrap_ci=False)
        attempt('original_resolution_errorbars',original_resolution_with_errorbars)
    else:
        notes.append({'analysis':'original_resolution_errorbars',
                      'reason':'Cross-configuration prediction event order not identical'})
    # The original cell 756/757 also produces best-score ranked histograms.
    ranked=d0.sort_values('d0')
    subset=ranked[ranked.data_labels!='Standard FastCalo'].head(10)
    if len(subset):
        names=sorted(subset.data_labels.to_list(),key=rmse_for,reverse=True)
        attempt('original_ranked_histogram',lambda:op.comparison_hist_boxplot_erros(
            [base]+[data[k].residual_calib_MeV.to_numpy(float) for k in names],
            ['Standard FastCalo']+names,op.full_data_text_label,
            xtextlabel=r'$E_{true}-E_{calib}$ (MeV)'))
    # Include archived numerical summary tables, with per-model events, in
    # their original units. The notebook presents these as DataFrames, not images.
    summary_rows=[]
    for lab,d in data.items():
        e=d.residual_calib_MeV.to_numpy(float)
        raw=d.residual_fast_MeV.to_numpy(float)
        summary_rows.append({'configuration':lab,'n_events':len(e),
            'RMSE_MeV':float(np.sqrt(np.mean(e*e))),
            'IQR_MeV':float(np.quantile(e,.75)-np.quantile(e,.25)),
            'd0_MeV':float(np.hypot(np.sqrt(np.mean(e*e)),np.quantile(e,.75)-np.quantile(e,.25))),
            'paired_mean_difference_MeV':float(np.mean(e-raw))})
    summary=pd.DataFrame(summary_rows).sort_values('d0_MeV')
    summary.to_csv(out/'computed_global_performance.csv',index=False)
    # Original notebook cells 733–735: D0 ranking. Numeric inference times
    # must come from the saved archives; never substitute manuscript numbers.
    candidate_paths=[parent/'artifacts'/'metadata'/'prediction_times.csv'
                     for parent in [out,*out.parents]]
    timing_csv=next((p for p in candidate_paths if p.is_file()),None)
    if timing_csv is not None:
        try:
            from .analysis_tables import compute_D0, configuration_feature_count, archived_feature_counts
            timing=pd.read_csv(timing_csv)
            perf=summary.rename(columns={'configuration':'configuration',
                    'RMSE_MeV':'rmse_MeV','IQR_MeV':'iqr_MeV'}).copy()
            feature_counts=archived_feature_counts(bundles)
            perf['F']=perf.configuration.map(lambda name: feature_counts.get(name,configuration_feature_count(name)))
            required_names=set(summary.configuration)
            timing=timing[timing.configuration.isin(required_names)].copy()
            # The author's comparison_metric_all routine expects the Standard
            # FastCalo reference (label Standard FastCalo:0) to be present.
            # Derive its RMSE/IQR from the same stored test-event residuals and
            # use zero input features / zero regressor prediction time.
            # This is the baseline method, not a fitted estimator.
            perf=perf.merge(timing[['configuration','time_ms_per_event']],
                            on='configuration',how='inner',validate='one_to_one')
            if perf.empty:raise ValueError('No archived prediction timing matches a saved configuration')
            perf=perf[perf.F.gt(0)].copy()
            if perf.empty:raise ValueError('No verified feature counts for timed configurations')
            baseline=pd.DataFrame([{
                'configuration':'Standard FastCalo:0',
                'rmse_MeV':float(np.sqrt(np.mean(base**2))),
                'iqr_MeV':float(np.quantile(base,.75)-np.quantile(base,.25)),
                'F':0,
                'time_ms_per_event':0.0,
                'd0_MeV':float(np.hypot(np.sqrt(np.mean(base**2)),
                                           np.quantile(base,.75)-np.quantile(base,.25)))
            }])
            score=pd.concat([perf,baseline],ignore_index=True)
            if 'd0_MeV' not in score:
                score['d0_MeV']=np.hypot(score.rmse_MeV,score.iqr_MeV)
            else:
                score['d0_MeV']=np.hypot(score.rmse_MeV,score.iqr_MeV)
            cols=['iqr_MeV','rmse_MeV','F','time_ms_per_event']
            X=score[cols].to_numpy(dtype=float)
            mn=X.min(axis=0); span=X.max(axis=0)-mn
            Xnorm=np.divide(X-mn,span,out=np.zeros_like(X),where=span>0)
            score['D0']=np.sqrt(np.sum(Xnorm**2,axis=1))
            score['Method-Selection']=score.configuration
            score['d0 (GeV)']=score.d0_MeV/1000.0
            score=score.sort_values('D0').reset_index(drop=True)
            score.to_csv(out/'computed_D0_ranking.csv',index=False)
            attempt('original_D0_ranking',lambda:op.comparison_metric_all(
                 score,op.full_data_text_label,rank='D0',show='D0'))
            attempt('original_D0_ranked_d0',lambda:op.comparison_metric_all(
                 score,op.full_data_text_label,rank='D0',show='d0'))
        except Exception as exc:
            notes.append({'analysis':'original_D0_ranking','reason':f'{type(exc).__name__}: {exc}'})
    else:notes.append({'analysis':'original_D0_ranking',
         'reason':'Numerical original prediction times are required; extract *_noSEG_times.pkl'})
    # Further result figures executed after "Comparison Between All Methods"
    # in the original notebook (cells 702, 708, 774 and 775).
    metric_plot=d0.copy().reset_index(drop=True)
    attempt('original_all_d0_comparison',lambda:op.comparison_metric(
        metric_plot,op.full_data_text_label))
    if timing_csv is not None and timing_csv.is_file():
        time_records=pd.read_csv(timing_csv)
        if {'configuration','time_ms_per_event'}.issubset(time_records.columns):
            time_records=time_records[time_records.configuration.isin(data)].copy()
            time_records=time_records.dropna(subset=['time_ms_per_event'])
            if not time_records.empty:
                time_records=pd.DataFrame({
                    'Method-Selection':time_records.configuration.to_numpy(),
                    'Processing-time':time_records.time_ms_per_event.to_numpy(dtype=float)/1000.0})
                time_records=pd.concat([time_records,pd.DataFrame([
                    {'Method-Selection':'Standard FastCalo','Processing-time':0.0}])],
                    ignore_index=True)
                attempt('original_all_prediction_times',lambda:op.comparison_times(
                    time_records,op.full_data_text_label))
    historical_metrics=[]
    required_stats={'Evaluation-Method','Feature-Selection-Method',
        'F-H0: f-value (0.05)','F-H0: f-critical (0.05)',
        'T-H0: t-value (0.05)','T-H0: t-critical (0.05)',
        'F-H1: IC[Errcalib_var/Errfast_var; 0.95]',
        'T-H1: IC[Errfast_mean - Errcalib_mean; 0.95]'}
    for method,bundle in bundles.items():
        tab=bundle.get(f'{method}_noSEG_performance_metrics')
        if isinstance(tab,pd.DataFrame) and required_stats.issubset(tab.columns):
            historical_metrics.append(tab.copy())
    if historical_metrics:
        stats_table=pd.concat(historical_metrics,ignore_index=True)
        if 'd0 (MeV)' in stats_table:
            stats_table['d0 (MeV)']=pd.to_numeric(stats_table['d0 (MeV)'],errors='coerce')
            stats_table=stats_table.sort_values('d0 (MeV)',na_position='last')
        stats_table=stats_table.loc[stats_table['Feature-Selection-Method'].astype(str)
                                  .ne('Standard FastCalo')].head(10).copy()
        for c in ('F-H0: f-value (0.05)','F-H0: f-critical (0.05)',
                  'T-H0: t-value (0.05)','T-H0: t-critical (0.05)'):
            stats_table[c]=pd.to_numeric(stats_table[c],errors='coerce')
        stats_table=stats_table.dropna(subset=[
            'F-H0: f-value (0.05)','F-H0: f-critical (0.05)',
            'T-H0: t-value (0.05)','T-H0: t-critical (0.05)'])
        if not stats_table.empty:
            stats_table.to_csv(out/'original_statistical_tests.csv',index=False)
            attempt('original_statistics_intervals_vertical',lambda:op.plota_stats_tests_intervals(
                stats_table.copy(),op.full_data_text_label))
            attempt('original_statistics_intervals_horizontal',lambda:op.plota_stats_tests_intervals_h(
                stats_table.copy(),op.full_data_text_label))
    # Original source-data diagnostics: notebook cells 118 and 157. These need
    # the Lorenzetti CSV; saved predictions alone cannot reproduce them.
    if article=='tns' and data_csv is not None:
        try:
            from .original_data_plots import draw_benchmark_diagnostics
            results.extend(draw_benchmark_diagnostics(data_csv,out))
        except Exception as exc:
            notes.append({'analysis':'original_benchmark_diagnostics',
                          'reason':f'{type(exc).__name__}: {exc}'})
    elif article=='tns':
        notes.append({'analysis':'original_benchmark_diagnostics',
                      'reason':'Input Lorenzetti CSV required for original source-data diagnostics'})
    # Feature relevance is calculated from the D70 dataset only, using the
    # source representation operators. It cannot be recovered from predictions.
    if article=='tns' and data_csv is not None:
        try:
            from .feature_relevance import draw_feature_relevance
            p=draw_feature_relevance(data_csv,out/'stage1_feature_relevance.png')
            results.append(p)
        except Exception as exc:
            notes.append({'analysis':'stage1_feature_relevance',
                          'reason':f'{type(exc).__name__}: {exc}'})
    mapping={
      'tns':{
        19:('original_energy_histogram','original_fast_vs_truth'),
        20:('stage1_feature_relevance',),
        21:('original_hist_boxplot_',),22:('original_segmented_',),
        23:('original_segmented_',),24:('original_d0_ranking',),25:('original_qq',)},
      'measurement':{
        26:('original_hist_boxplot_',),27:('original_D0_ranking','original_d0_ranking'),
        28:('original_segmented_',),29:('original_segmented_',),
        30:('original_segmented_boxplot',),31:('original_resolution_',),
        32:('original_EAR_',),33:('original_scatter_',),34:('original_qq_',)},
      'information_fusion':{
        35:('original_hist_boxplot_',),36:('original_D0_ranking','original_d0_ranking'),
        37:('original_Fig37',),38:('original_Fig38',),
        39:('original_segmented_',),40:('original_segmented_',),
        41:('original_segmented_boxplot',),42:('original_resolution_',),
        43:('original_EAR_',),44:('original_scatter_',),45:('original_qq_',)},
    }[article]
    inventory=pd.DataFrame([{
        'thesis_figure':fig,
        'source_function_groups':' | '.join(tags),
        'output_files':' | '.join(p.name for p in results if any(p.name.startswith(tag) for tag in tags)),
        'status':'generated' if any(p.name.startswith(tag) for p in results for tag in tags) else 'missing_input',
    } for fig,tags in mapping.items()])
    inventory.to_csv(out/'publication_figure_coverage.csv',index=False)
    return {'figure_paths':results,'notes':notes,'best':best,'methods':len(data),'ranking':d0,'computed_summary':summary,'figure_coverage':inventory}
