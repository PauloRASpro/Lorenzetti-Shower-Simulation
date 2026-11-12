#!/usr/bin/env python3
"""Generate article analyses with the plotting functions of the experimental notebook.

Examples:
  python scripts/reproduce_article.py --article tns --list
  python scripts/reproduce_article.py --article measurement --trust-pickle
  python scripts/reproduce_article.py --article information_fusion --trust-pickle

Pickle deserialization is restricted to archives whose origin you trust.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lorenzetti_repro.legacy_artifacts import ARTICLE_METHODS,catalog
from lorenzetti_repro.article_workflow import load_article_artifacts,historic_method_tables
from lorenzetti_repro.original_replay import replay_original_figures

def reproduce(article,folder=None,output=None,trust_pickle=False):
    path=Path(folder or ROOT/'artifacts'/'pickles')
    if not trust_pickle:
        inventory=catalog(path,checksum=False)
        print(json.dumps(inventory,indent=2,ensure_ascii=False))
        print('Use --trust-pickle only for research archives from a trusted source.')
        return {'files':len(inventory)}
    _, inventory,bundles,frames=load_article_artifacts(article,ROOT,trust_local_pickle=True,pickle_dir=path)
    dest=Path(output or ROOT/'results'/'replayed'/article)
    tables,files=historic_method_tables(bundles,dest/'tables')
    report=replay_original_figures(article,frames,bundles,dest/'original_figures')
    summary={'article':article,'archive':str(path),'methods':list(bundles),
             'event_configurations':list(frames),
             'generated_figures':[str(p) for p in report['figure_paths']],
             'original_tables':list(tables),'notes':report['notes']}
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'replay_manifest.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
    print(f'Generated {len(report["figure_paths"])} figures via original routines; {len(tables)} tables')
    return summary

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--article',required=True,choices=sorted(ARTICLE_METHODS))
    ap.add_argument('--folder');ap.add_argument('--output');ap.add_argument('--list',action='store_true')
    ap.add_argument('--trust-pickle',action='store_true')
    a=ap.parse_args()
    if a.list:
        print('Study:',a.article,'Model families:',', '.join(ARTICLE_METHODS[a.article]))
        print('Original drawing functions: comparison_hist_boxplot_erros, comparison_metric_all, '
              'integrated_plots_in_ETA_ET_geral_with_ranges, boxplot_comparison_improved, '
              'plotar_resolucao, plotar_resolucao_multipla, qq_plot, lambda_analysis')
        return
    reproduce(a.article,a.folder,a.output,a.trust_pickle)
if __name__=='__main__':main()
