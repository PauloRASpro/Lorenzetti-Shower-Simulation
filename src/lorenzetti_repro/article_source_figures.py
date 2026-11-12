"""Reproduce original pre-training benchmark figures with exact research plotting statements.

All rendering code is sourced from the author's experiment notebook: original_article_cells.
No PDF figures, original notebook image outputs, or published numerical metrics are loaded.
"""
from __future__ import annotations
from pathlib import Path
from statistics import mean, stdev
import numpy as np
import pandas as pd
from .original_article_cells import SOURCE_CELLS
from .original_replay import collect_shows
from .original_plotting import plota_data_label, plota_data_label_ax
from .original_figure_environment import source_figure_label


def generate_source_diagnostics(data_csv, output, *, feature_map=True):
    """Return {figure: [paths]}, along with exact source-cell provenance and errors.

    Plot-source statements retain the original figure sizes, labels, fonts, legends,
    axes and all other Matplotlib display arguments. Dataset calculations follow
    the source notebook where compatible with the thesis' D70-only representation.
    """
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    full_data=pd.read_csv(data_csv)
    for key in ('mc_match_et','cluster_truthet','cluster_trutheta'):
        if key not in full_data: raise KeyError(f'Original Lorenzetti CSV lacks {key!r}')
    full_data=full_data.replace([np.inf,-np.inf],np.nan).dropna(subset=['mc_match_et','cluster_truthet','cluster_trutheta']).copy()
    full_data=full_data.loc[full_data.cluster_truthet!=0].copy()
    full_data['alfa']=full_data.mc_match_et/full_data.cluster_truthet
    def norm_data(data):
        a=np.asarray(data,float)
        return (a-np.mean(a))/stdev(a)
    env={'plt':plt,'np':np,'pd':pd,'full_data':full_data,
         'plota_data_label':plota_data_label,'plota_data_label_ax':plota_data_label_ax,'full_data_text_label':source_figure_label(),
         'norm_data':norm_data,'stdev':stdev,'mean':mean}
    outputs={};failures={}
    def run(tag, ids, prep=None):
        try:
            ctx=env.copy()
            if prep is not None: prep(ctx)
            with collect_shows(output,tag) as files:
                for i in ids:exec(SOURCE_CELLS[str(i)],ctx,ctx)
            if not files:raise RuntimeError('Original code emitted no plot')
            outputs[tag]=list(files)
        except Exception as e:
            failures[tag]=type(e).__name__+': '+str(e)
    # The corresponding primary-source cells control all figure typography and geometry.
    run('original_true_fast_histogram',[120])
    run('original_true_fast_scatter',[118])
    def error_context(ctx):
        t=full_data.mc_match_et.to_numpy(float)
        f=full_data.cluster_truthet.to_numpy(float)
        ctx.update(erro=t-f, erro_norm=(t-f)/t, erro_alfa=t/f)
    run('original_standardized_error_histogram',[75,76],error_context)
    # RSS goodness-of-fit (paper Fig. 5(b) / Fig. 6(b)). The original research
    # code requires distfit; do not replace its fit values with invented scores.
    def fit_context(ctx):
        error_context(ctx)
    run('original_error_distribution_fit',[60,64,68,72,74],fit_context)
    # True-calibration-factor boxplots use exactly the original bin-construction code.
    run('original_true_alpha_eta',[206,207,208,209])
    run('original_true_alpha_energy',[211,212,213,214])
    if all(str(i) in full_data for i in range(100)):
        run('original_100_ring_boxplot',[163,164])
    else:
        failures['original_100_ring_boxplot']='The original 100 ring columns (0..99) are absent.'
    if feature_map and all(str(i) in full_data for i in range(100)):
        try:
            from .feature_relevance import feature_scores
            scores=feature_scores(data_csv)
            # Original cells 195–199 build min-max-scaled relevance for visualization.
            def scaled(v):
                v=np.asarray(v,float)
                d=v.max()-v.min()
                return (v-v.min())/d if d else np.zeros_like(v)
            import seaborn as sns
            context=dict(env, sns=sns,
                 df_dict_methods_pca_ica_corr_mut=pd.DataFrame({
                    'PCA':np.sort(scaled(scores['PCA']))[::-1],
                    'ICA':np.sort(scaled(scores['ICA']))[::-1],
                    'CORR':scaled(scores['CORR']),
                    'MUTI':scaled(scores['MUTI'])}))
            with collect_shows(output,'original_feature_relevance') as files:
                exec(SOURCE_CELLS['199'],context,context)
            if not files: raise RuntimeError('Heatmap emitted no figure')
            outputs['original_feature_relevance']=list(files)
        except Exception as e:
            failures['original_feature_relevance']=type(e).__name__+': '+str(e)
    else:
        failures['original_feature_relevance']='The source CSV and 100 rings are required.'
    return {'paths':outputs,'failures':failures,'source_cells':{
        'original_true_fast_histogram':[120], 'original_true_fast_scatter':[118],
        'original_standardized_error_histogram':[75,76],
        'original_error_distribution_fit':[60,64,68,72,74],
        'original_true_alpha_eta':[206,207,208,209],
        'original_true_alpha_energy':[211,212,213,214],
        'original_100_ring_boxplot':[163,164], 'original_feature_relevance':[195,199]}}
