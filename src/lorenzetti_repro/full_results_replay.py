"""Archived experimental comparisons generated with the notebook's plotting code.

Each figure uses the functions or inline plotting statements in the author's
experimental notebook.  Data are exclusively derived from event predictions,
original aggregated tables, and measured timing artifacts.
"""
from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .legacy_artifacts import event_values
from . import original_plotting as op
from .original_result_cells import ORIGINAL_RESULT_CELLS
from .original_replay import collect_shows
from .original_figure_environment import source_figure_label
from .analysis_tables import archived_feature_counts, configuration_feature_count


def _iqr(a):
    return float(np.percentile(a,75)-np.percentile(a,25))


def _method_and_rep(label):
    model,sep,rep=str(label).partition(':')
    return model.strip(), rep.strip() if sep else ''


def archived_comparison_tables(frames, bundles, timing_csv=None):
    """Build original comparison fields from the stored per-event results."""
    if not frames:return {},pd.DataFrame()
    methods=[]
    first=None
    for key,frame in frames.items():
        v=event_values(frame)
        if first is None:first=v
        residual=v.residual_calib_MeV.to_numpy(float)
        if not len(residual) or not np.isfinite(residual).all():continue
        method,rep=_method_and_rep(key)
        rmse=float(np.sqrt(np.mean(residual**2)));iqr=_iqr(residual)
        methods.append(dict(method=method,useRings=rep,data_erros=residual,data_labels=key,
                            iqr=iqr,rmse=rmse,d0=float(np.hypot(iqr,rmse))))
    if first is None:return {},pd.DataFrame()
    baseline=first.residual_fast_MeV.to_numpy(float)
    if not len(baseline):return {},pd.DataFrame()
    basermse=float(np.sqrt(np.mean(baseline**2)));baseiqr=_iqr(baseline)
    baseline_d0=float(np.hypot(basermse,baseiqr))
    methods.append(dict(method='Standard FastCalo',useRings=0,data_erros=baseline,
                        data_labels='Standard FastCalo',iqr=baseiqr,rmse=basermse,d0=baseline_d0))
    errors=pd.DataFrame(methods).sort_values('d0').reset_index(drop=True)
    times=pd.DataFrame()
    if timing_csv and Path(timing_csv).is_file():
        times=pd.read_csv(timing_csv)
        if {'configuration','time_ms_per_event'}.issubset(times.columns):
            times=times[['configuration','time_ms_per_event']].copy()
            times['configuration']=times.configuration.astype(str).str.strip()
            times=times.drop_duplicates('configuration')
        else:times=pd.DataFrame()
    counts=archived_feature_counts(bundles)
    table=[]
    for row in methods:
        label=row['data_labels'];n=counts.get(label,configuration_feature_count(label))
        if label=='Standard FastCalo':n=1
        table.append({'Method-Selection':'Standard FastCalo:0' if label=='Standard FastCalo' else label,
                      'configuration':label,'IQR (MeV)':row['iqr'], 'RMSE (MeV)':row['rmse'],
                      'd0 (GeV)':row['d0']/1000,'d0':row['d0'], 'Number of Features':n})
    summary=pd.DataFrame(table)
    if not times.empty:
        summary=summary.merge(times,how='left',on='configuration',validate='one_to_one')
    else:summary['time_ms_per_event']=np.nan
    summary.loc[summary.configuration=='Standard FastCalo','time_ms_per_event']=0.0
    # Original notebook normalizes these four quantities within the comparison.
    valid=(summary['Number of Features']>0)&summary['time_ms_per_event'].notna()
    complete=summary.loc[valid].copy()
    if len(complete)>=2:
        matrix=complete[['IQR (MeV)','RMSE (MeV)','time_ms_per_event','Number of Features']].to_numpy(float)
        mn=matrix.min(axis=0);ptp=np.ptp(matrix,axis=0)
        z=np.divide(matrix-mn,ptp,out=np.zeros_like(matrix),where=ptp>0)
        complete['D0']=np.sqrt(np.sum(z*z,axis=1))
        complete['Time (ms)']=complete['time_ms_per_event']
    return {'original_error_results':errors,'global_performance':summary,'complete_efficiency':complete},errors


def generate_full_original_analyses(article,frames,bundles,output,*,timing_csv=None):
    """Generate non-test research plots in the notebook's comparison section.

    This is intentionally distinct from the journal-figure subset. Each
    recovered analysis is displayed by the corresponding article notebook.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    tables,errors=archived_comparison_tables(frames,bundles,timing_csv)
    if errors.empty:return {'paths':{},'errors':{'comparisons':'No archived per-event results'},'tables':{}}
    source_label=source_figure_label()
    outputs={};problems={}
    def emit(tag,fn):
        try:
            with collect_shows(output,tag) as files:fn()
            if not files:raise RuntimeError('Plotting returned no figure')
            outputs[tag]=list(files)
        except (Exception) as exc:
            problems[tag]=f'{type(exc).__name__}: {exc}'
    # Original source cells 702 and 705: d0 rankings.
    e=errors.copy().reset_index(drop=True)
    emit('comparison_metric_cell702',lambda:op.comparison_metric(e,source_label))
    # Both the baseline and all archived configurations are included.
    emit('comparison_metric_all_cell705',lambda:op.comparison_metric_all(e,source_label,rank='d0',show='d0'))
    # Original source cell 749: top ten by d0, with a common FastCalo residual.
    calibrators=e[e.data_labels!='Standard FastCalo'].head(10)
    baseline=e[e.data_labels=='Standard FastCalo']
    if len(calibrators):
        residuals=list(calibrators.data_erros)+list(baseline.data_erros)
        labels=list(calibrators.data_labels)+list(baseline.data_labels)
        emit('top_ten_residuals_cell749',lambda:op.comparison_hist_boxplot_erros(
             residuals,labels,source_label,xtextlabel=r'$E_{true} - E_{calib}$'))
        # Original source cell 767, unmodified graphical statements.
        if len(calibrators)>=10:
            env={'np':np,'plt':plt,'All_NoSeg_erros_data':e,
                 'full_data_text_label':source_label}
            emit('rmse_iqr_quadrant_cell767',lambda:exec(ORIGINAL_RESULT_CELLS['767'],env,env))
    # Original source cell 746 compares different models at the SAME feature representation.
    for rep,subset in errors[errors.useRings!=0].groupby('useRings',sort=True):
        if subset.empty:continue
        subset=subset.sort_values('rmse',ascending=False)
        vals=list(subset.data_erros)+list(baseline.data_erros)
        labels=list(subset.data_labels)+list(baseline.data_labels)
        tag='comparison_by_representation_'+re.sub(r'[^A-Za-z0-9]+','_',str(rep)).strip('_')
        emit(tag,lambda vals=vals,labels=labels,rep=rep:op.comparison_hist_boxplot_erros(
            vals,labels,source_label,xtextlabel=r'$E_{true} - E_{calib}$ (MeV), '+str(rep)))
    # Original source cell 747 compares all archived models for each feature family.
    for family in ('Rings','CORR','MUTI','PCA','ICA'):
        subset=errors[errors.useRings.astype(str).str.contains(family,regex=False,case=False)].sort_values('rmse',ascending=False)
        if subset.empty:continue
        vals=list(subset.data_erros)+list(baseline.data_erros)
        labels=list(subset.data_labels)+list(baseline.data_labels)
        emit(f'comparison_by_family_{family.lower()}_cell747',lambda vals=vals,labels=labels,family=family:
             op.comparison_hist_boxplot_erros(vals,labels,source_label,
                     xtextlabel=r'$E_{true} - E_{calib}$ (MeV), '+family))
    # Original source cells 708,713,722,733-735,756-757 use measured times and
    # actual selected-feature counts; outputs requiring them are not fabricated.
    t=tables['complete_efficiency']
    timing_table=tables['global_performance']
    if len(timing_table)>0:
        measured=timing_table.loc[(timing_table.configuration!='Standard FastCalo') & timing_table.time_ms_per_event.notna()].copy()
        if not measured.empty:
            ttable=pd.DataFrame({'Method-Selection':measured['Method-Selection'].to_numpy(),
                                'Processing-time':measured['time_ms_per_event'].to_numpy(float)/1000.0})
            emit('processing_times_cell708',lambda:op.comparison_times(ttable,source_label))
            # Original cell 713: compare d0 and per-event prediction time.
            times_by_label={r['configuration']:r['time_ms_per_event'] for _,r in measured.iterrows()}
            timed=e.copy()
            timed['configuration']=timed.data_labels
            timed['time_ms']=timed.configuration.map(times_by_label).fillna(0.0)
            if timed.loc[timed.method!='Standard FastCalo','time_ms'].gt(0).sum()>=10:
                env={'np':np,'pd':pd,'plt':plt,'full_data_text_label':source_label,
                  'All_NoSeg_erros_data':timed,'d0_method':list(timed.method),
                  'd0_selection':list(timed.useRings),
                  'd0_metric':list(timed.d0/1000.0),
                  'All_noSeg_processing_times':pd.DataFrame({
                     'Method':list(timed.method),'Selection':list(timed.useRings),
                     'Processing-time':list(timed.time_ms)})}
                emit('d0_vs_prediction_time_cell713',lambda:exec(ORIGINAL_RESULT_CELLS['713'],env,env))
            # Original cell 722: use the actual, archived number of features.
            counts_by_label={r['configuration']:r['Number of Features'] for _,r in
                             timing_table.iterrows()}
            counted=errors.copy()
            counted['n_features']=counted.data_labels.map(counts_by_label)
            if counted.n_features.gt(0).sum()>=10:
                feat_df=counted[counted.n_features.gt(0)].copy()
                feat_df['Features']=feat_df.n_features.map(lambda n:list(range(int(n))))
                env={'np':np,'pd':pd,'plt':plt,'full_data_text_label':source_label,
                     'table_results_sorted_features':pd.DataFrame({
                          'Method':feat_df.method,'Selection':feat_df.useRings,
                          'Features':feat_df.Features}),
                     'd0_method':list(counted.method),'d0_selection':list(counted.useRings),
                     'd0_metric':list(counted.d0/1000.0)}
                # Cell 722 uses integer-index loops into d0_metric, so the full
                # ordered archive must be represented in the feature lookup.
                emit('d0_vs_feature_count_cell722',lambda:exec(ORIGINAL_RESULT_CELLS['722'],env,env))
    if len(t)>1 and 'D0' in t:
        qualified=t.sort_values('D0').reset_index(drop=True)
        if len(qualified)>10 and (qualified['Method-Selection']=='Standard FastCalo:0').any():
            emit('D0_ranked_cell733',lambda:op.comparison_metric_all(qualified,source_label,rank='D0',show='D0'))
            emit('d0_ordered_by_D0_cell734',lambda:op.comparison_metric_all(qualified,source_label,rank='D0',show='d0'))
            emit('D0_ordered_by_d0_cell735',lambda:op.comparison_metric_all(qualified,source_label,rank='d0',show='D0'))
        else:problems['D0_ranking']='D0 requires at least ten archived configurations including the baseline and measured times/features'
        # Original histogram ordering by the archived ranking (cells 756-757).
        for rank in ('d0 (GeV)','D0'):
            top=qualified.sort_values(rank).head(10)
            keys=[s.replace('Standard FastCalo:0','Standard FastCalo') for s in top['Method-Selection']]
            hit=e.set_index('data_labels').reindex(keys).dropna(subset=['data_erros'])
            if len(hit):
                vals=list(hit.data_erros);labs=list(hit.index)
                emit('top_ten_by_'+('D0' if rank=='D0' else 'd0')+'_cells756_757',
                     lambda vals=vals,labs=labs:op.comparison_hist_boxplot_erros(
                         vals,labs,source_label,xtextlabel=r'$E_{true} - E_{calib}$ (MeV)'))
    else:problems['timing_based_comparisons']='No common measured times and documented feature counts'
    # Original cell 789: paired QQ normality of original, calibrated and difference.
    anchor=e.loc[e.data_labels!='Standard FastCalo'].iloc[0]
    calibrated=np.asarray(anchor.data_erros,float)
    fast=np.asarray(baseline.iloc[0].data_erros,float)
    if len(fast)==len(calibrated):
        emit('paired_difference_qq_cell789',lambda:op.qq_plot(calibrated-fast,'Difference in paired residuals'))
    # Original cell 805 applies paired normality/QQ to the selected research
    # comparison (four representatives), rather than the entire grid search.
    selected_models={
      'tns':[('BDT','PCA-10'),('BDT','CORR-90'),('BDT','MUTI-90'),('BDT','ICA-10')],
      'measurement':[('RFBDT','Rings[1:100]'),('BDT','ICA-5'),('BDT','PCA-10'),('BDT','Rings[1:100]')],
      'information_fusion':[('GNN','Rings[1:100]'),('DNN','Rings[1:100]'),('DNN','ICA-10'),('DNN','CORR-10')]
    }
    selected_keys=[]
    for model,rep in selected_models[article]:
        key=next((label for label in frames if
            label.partition(':')[0].strip().replace(' no-Bin','').strip()==model
            and label.partition(':')[-1].strip()==rep),None)
        if key is not None:selected_keys.append(key)
    for j,key in enumerate(selected_keys):
        frame=frames[key]
        if len(key)>160:continue
        d=event_values(frame);r=d.residual_calib_MeV.to_numpy(float);f=d.residual_fast_MeV.to_numpy(float)
        if len(r)!=len(f):continue
        tag=f'paired_qq_{j+1:03d}_cell805'
        emit(tag,lambda r=r,f=f,key=key:(
            op.qq_plot(f,'Standard FastCalo: '+key),
            op.qq_plot(r,'Calibrated: '+key),
            op.qq_plot(r-f,'Paired difference: '+key)))
    # Source cell 816: error-bar resolution versus true energy, using
    # the original 20 equal-width energy bins and at least 100 events per bin.
    if selected_keys:
        ordered=[key for key in selected_keys if key in frames]
        candidate=[event_values(frames[key]) for key in ordered]
        same=(all(len(v)==len(candidate[0]) and
             np.array_equal(v.E_true_MeV,candidate[0].E_true_MeV) and
             np.array_equal(v.E_fast_MeV,candidate[0].E_fast_MeV)
             for v in candidate[1:]))
        if same:
            Etrue=candidate[0].E_true_MeV.to_numpy(float)
            methods={'Standard FastCalo':candidate[0].E_fast_MeV.to_numpy(float)}
            for key,v in zip(ordered,candidate):
                methods[key]=v.E_calib_MeV.to_numpy(float)
            try:
                res=op.compute_resolution_by_energy(Etrue,methods,n_bins=20,
                    bin_strategy='width',min_count=100,robust=False,
                    use_full_SE_for_R=True,bootstrap_B=None)
                if res:
                    emit('resolution_with_uncertainty_cell816',
                         lambda:op.plot_resolution_curves(res,
                            methods_order=list(methods),show_errorbars=True,use_bootstrap_ci=False))
                    pd.DataFrame(res).to_csv(output/'resolution_with_uncertainty_cell816.csv',index=False)
            except Exception as exc:problems['resolution_with_uncertainty_cell816']=f'{type(exc).__name__}: {exc}'
        else:problems['resolution_with_uncertainty_cell816']='Selected models have different holdout event orders'
    saved_tables={}
    for key,frame in tables.items():
        if isinstance(frame,pd.DataFrame) and not frame.empty:
            name=key+'.csv';frame.drop(columns=['data_erros'],errors='ignore').to_csv(output/name,index=False)
            saved_tables[key]=output/name
    return {'paths':outputs,'errors':problems,'tables':saved_tables}
