#!/usr/bin/env python3
"""Thesis-aligned *new* training run of one documented article configuration.

Default notebooks never call this script (RUN_EXPERIMENTS=False). No source
paper number is presented as a result from this script until independently
verified against the frozen dataset, splits and environment.
"""
import argparse,json,sys,datetime
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lorenzetti_repro.dataset import (read_dataset,find_dataset,RINGS,TRUE,FAST,split_70_30,alpha_target,sha256_file,EXPECTED_EVENTS)
from lorenzetti_repro.representations import Representation
from lorenzetti_repro.model_runner import fit_predict_model
from lorenzetti_repro.selection import nested_select_energy
from lorenzetti_repro.metrics import assess,shared_ear,delta_lambda_90

ROOT=Path(__file__).resolve().parents[1]
STAGE_NAMES={'1':'IEEE TNS','2':'Measurement','3':'Information Fusion'}

def get_params(stage,model,rep,mode):
    directory=ROOT/'configs'/('stage'+stage)
    if mode=='nested':
        if stage=='1':raise ValueError('Stage I uses the historical protocol, NOT 5x5 nested CV')
        fp=directory/(model+'_search.json')
        if not fp.exists():
            raise ValueError(f'No source-supported search grid for {model}, stage {stage}.')
        return json.loads(fp.read_text(encoding='utf-8'))['grid'],fp
    if stage=='1':
        fp=directory/'documented_candidate_configurations.json'
        cfg=json.loads(fp.read_text(encoding='utf-8'))
        if model=='bdt-bin':model='bdt'
        if model not in cfg['models']:raise ValueError(f'Unknown Stage I model: {model}')
        return cfg['models'][model],fp
    normalized=rep.lower().replace('[','').replace(']','').replace(':','')
    if normalized in ('full','rings[1:100]'):normalized='rings100'
    fp=directory/(model+'_'+normalized+'.json')
    if not fp.exists():
        raise ValueError(f'No archived final configuration for {model}/{rep} in stage {stage}. '
                         'The published result is included in results/reference; do not fabricate missing final hyperparameters. '
                         f'See {directory}.')
    return json.loads(fp.read_text(encoding='utf-8'))['parameters'],fp


def run(stage,model,representation,data,out,mode,split_seed=0,allow_size_mismatch=False):
    if stage not in STAGE_NAMES:raise ValueError('Stage must be 1/2/3')
    if model=='gnn' and representation.lower()!='rings100':raise ValueError('The published graph model uses the full 100 rings')
    if stage!='3' and model in ('dnn','gnn'):raise ValueError('DNN and GNN are Stage III models')
    if stage=='1' and model in ('rf','rfbdt','bdtrf','rfbdts'):raise ValueError('Stage I uses conventional models only')
    path=find_dataset(root=ROOT/'data',explicit=data)
    df=read_dataset(path)
    if len(df)!=EXPECTED_EVENTS and not allow_size_mismatch:
        raise ValueError(f'{len(df)} events found, expected {EXPECTED_EVENTS}. To run a NON-historical subset use --allow-size-mismatch.')
    idx_dev,idx_test=split_70_30(df,random_state=split_seed)
    X=df[RINGS].to_numpy(dtype=float);truth=df[TRUE].to_numpy(dtype=float);fast=df[FAST].to_numpy(dtype=float)
    alpha=alpha_target(df)
    # g_ML is established using D70 and NEVER sees D30 fitting information.
    g=Representation(representation,seed=0,stage=int(stage)).fit(X[idx_dev],truth[idx_dev])
    XD=g.transform(X[idx_dev]);XT=g.transform(X[idx_test]);
    values,config_path=get_params(stage,model,representation,mode)
    nested=None
    if mode=='nested':
        nested=nested_select_energy(lambda a,b,c,p:fit_predict_model(model,a,b,c,p),
                                    XD,alpha[idx_dev],fast[idx_dev],truth[idx_dev],values)
        params=nested['selected_params']
    else:params=values
    if model=='bdt-bin':
        from lorenzetti_repro.models.stage1_segmented import MCSegmentedBDT
        from lorenzetti_repro.dataset import ETA
        if stage!='1':raise ValueError('The binned MC-only model belongs only to Stage I')
        if ETA not in df:raise ValueError('Missing cluster_trutheta for Stage I MC-only routing')
        etavec=df[ETA].to_numpy(dtype=float)
        fitted=MCSegmentedBDT(**params).fit(XD,alpha[idx_dev],etavec[idx_dev],truth[idx_dev])
        yh=fitted.predict(XT,etavec[idx_test],truth[idx_test])
    else:
        yh=fit_predict_model(model,XD,XT,alpha[idx_dev],params)
    metrics=assess(truth[idx_test],fast[idx_test],yh)
    grid,curve_cal,curve_fast=shared_ear(truth[idx_test],fast[idx_test]*yh,fast[idx_test])
    metrics['delta_lambda90_calibrated_mev']=delta_lambda_90(grid,curve_cal)
    metrics['delta_lambda90_baseline_mev']=delta_lambda_90(grid,curve_fast)
    from lorenzetti_repro.statistics import five_disjoint_subsets,paired_residual_checks
    subsets=five_disjoint_subsets(truth[idx_test],fast[idx_test],yh)
    if stage=='1':
        metrics['rmse_gev']=metrics['rmse']/1000.0
        metrics['iqr_gev']=metrics['iqr']/1000.0
        metrics['d0_gev']=metrics['d0']/1000.0
    out=Path(out)
    if not out.is_absolute():out=ROOT/out
    # This script is an independent single-model runner: never use a directory
    # containing historical pickle archives, and never overwrite a past run.
    protected=(ROOT/'artifacts'/'pickles').resolve()
    out_resolved=out.resolve()
    if out_resolved==protected or protected in out_resolved.parents:
        raise ValueError('Historical artifacts/pickles is read-only; specify a new unique output directory.')
    out.mkdir(parents=True,exist_ok=False)
    import csv
    if subsets:
        with (out/'five_disjoint_subset_metrics.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(subsets[0].keys()))
            writer.writeheader();writer.writerows(subsets)
    try:
        paired=paired_residual_checks(truth[idx_test],fast[idx_test],yh)
        (out/'new_run_paired_diagnostics.json').write_text(json.dumps(paired,indent=2),encoding='utf-8')
    except Exception as exc:
        (out/'new_run_paired_diagnostics_warning.txt').write_text(type(exc).__name__+': '+str(exc),encoding='utf-8')
    np.savez_compressed(out/'predictions.npz',event_row_ids=idx_test,et_true=truth[idx_test],et_fast=fast[idx_test],alpha_pred=yh)
    np.savez_compressed(out/'split_indices.npz',D70=idx_dev,D30=idx_test)
    report={'source_publication':STAGE_NAMES[stage],
            'scientific_status':'NEW RERUN; NOT VERIFIED REPRODUCTION OF PUBLISHED RESULTS',
            'generated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'stage':stage,'model':model,'representation':g.report(),
            'input_csv':str(path),'input_sha256':sha256_file(path),
            'input_rows':len(df),'D70':len(idx_dev),'D30':len(idx_test),
            'split_seed':split_seed,'historical_split_equality':'NOT VERIFIED',
            'config_path':str(config_path.relative_to(ROOT)),
            'config_status':('candidate; historical final unavailable' if stage=='1' else 'documented final') if mode!='nested' else 'new conditional nested selection',
            'selection_mode':mode,'parameters':params,'metrics_from_current_run':metrics,
            'truth_usage':'MC supervision/metrics only; never supplied directly as a regressor feature',
            'caveats':['no original split-index archives supplied',
                       'not certified equal to published historical run',
                       'no trained-model timing parity with publication'],
            'nested_selection':nested}
    (out/'run_manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,default=float),encoding='utf-8')
    print(json.dumps({'stage':stage,'model':model,'representation':representation,
                      'output':str(out),'status':report['scientific_status'],
                      'metrics':metrics},indent=2,ensure_ascii=False))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['1','2','3'],required=True)
    p.add_argument('--model',choices=['mlr','preg','dtree','bdt','bdt-bin','bspl','rf','rfbdt','bdtrf','rfbdts','dnn','gnn'],required=True)
    p.add_argument('--representation',required=True)
    p.add_argument('--data',help='Path to decompressed CSV; omission searches the repo data/ folder')
    p.add_argument('--mode',choices=['final-config','nested'],default='final-config')
    p.add_argument('--split-seed',type=int,default=0)
    p.add_argument('--out',required=True)
    p.add_argument('--allow-size-mismatch',action='store_true')
    a=p.parse_args()
    run(a.stage,a.model,a.representation,a.data,a.out,a.mode,a.split_seed,a.allow_size_mismatch)
if __name__=='__main__':main()
