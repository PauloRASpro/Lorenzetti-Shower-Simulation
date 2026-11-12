"""Core rendering must use the author's function bodies and real event arrays."""
import ast
from pathlib import Path
import numpy as np
import pandas as pd
from lorenzetti_repro import original_plotting
from lorenzetti_repro.original_replay import replay_original_figures

SRC=Path(__file__).resolve().parents[1]/'src'/'lorenzetti_repro'/'original_plotting.py'

def test_original_plotting_symbols_present():
    original={'comparison_hist_boxplot_erros','comparison_metric_all',
              'integrated_plots_in_ETA_ET_geral_with_ranges','boxplot_comparison_improved',
              'plotar_resolucao','plotar_resolucao_multipla','plot_resolution_curves',
              'lambda_analysis','qq_plot','plota_energy_alpha_scatter_por_metodo'}
    tree=ast.parse(SRC.read_text(encoding='utf8'))
    names={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
    assert original<=names


def test_historic_original_graphs_from_event_predictions(tmp_path):
    rng=np.random.default_rng(17);n=100
    true=rng.uniform(12000,70000,n);fast=true+rng.normal(0,900,n)
    alpha=true/fast+rng.normal(0,.007,n)
    label='BDT no-Bin: PCA-10'
    frame=pd.DataFrame({'method-selection':[label]*n,'ypred':alpha,'yte':true/fast,
        'cluster_truthet':fast,'mc_match_et':true,'cluster_trutheta':rng.uniform(-1,1,n),
        'calib_et':fast*alpha})
    result=replay_original_figures('tns',{label:frame},{},tmp_path)
    assert len(result['figure_paths'])>=7
    assert any('original_qq' in p.name for p in result['figure_paths'])
    assert any('original_EAR' in p.name for p in result['figure_paths'])
    assert all(p.exists() and p.stat().st_size>1000 for p in result['figure_paths'])


def test_no_other_method_is_silently_used_as_article_leader(tmp_path):
    n=30;true=np.linspace(15000,50000,n);fast=true+300
    label='BDT no-Bin: PCA-10'
    frame=pd.DataFrame({'method-selection':[label]*n,'ypred':true/fast,
                        'cluster_truthet':fast,'mc_match_et':true,'calib_et':true})
    result=replay_original_figures('information_fusion',{label:frame},{},tmp_path)
    assert not result['figure_paths']
    assert any('Expected GNN' in str(x) for x in result['notes'])
