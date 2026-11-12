#!/usr/bin/env python3
"""Extract small per-configuration timing tables from the author's own huge *_times.pkl.

Designed to run locally next to the original archives. Never copies trained
estimators into output. Unpickling requires an explicit trust flag; only load
personally controlled files. Run this once, then share only the compact CSV.
"""
from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lorenzetti_repro.legacy_artifacts import catalog,load_method_bundle
from lorenzetti_repro.analysis_tables import timing_from_archived_bundles

def export(folder,output,trust_pickle=False):
    if not trust_pickle:raise PermissionError('Use --trust-my-pickles for your personally verified archives')
    found=catalog(folder,checksum=False)
    methods=sorted({r['method'] for r in found if r['known'] and r['kind']=='noSEG_times'})
    if not methods:raise FileNotFoundError('No *_noSEG_times.pkl[.gz] found')
    bundle={}
    for m in methods:
        v=load_method_bundle(folder,m,trust_pickle=True,include_model_objects=True)
        key=f'{m}_noSEG_times'
        if key in v:bundle[m]={key:v[key]}
    table=timing_from_archived_bundles(bundle)
    if table.empty:raise ValueError('No compatible timing fields found in original DataFrames')
    if table.configuration.duplicated().any():
        raise ValueError('Timing labels not unique: resolve duplicate runs rather than silently keeping one')
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    table.to_csv(out,index=False)
    print(f'Exported {len(table)} timing configurations: {out}')
    return table

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--folder',default=str(ROOT/'artifacts'/'pickles'))
    p.add_argument('--output',default=str(ROOT/'artifacts'/'metadata'/'prediction_times.csv'))
    p.add_argument('--trust-my-pickles',action='store_true')
    a=p.parse_args();export(a.folder,a.output,a.trust_my_pickles)
