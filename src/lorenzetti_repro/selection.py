"""Conditional five-by-five hyperparameter selection, Chapter 5 §§5.7.2.

IMPORTANT: scores are computed from Etrue - Efast*alpha_pred, not the
alpha-domain error. g_ML is already fitted exclusively on full D70 and remains
fixed during the f_ML nested cross-validation (as the thesis specifies).
"""
from collections import Counter
from itertools import product
import json
import numpy as np
from sklearn.model_selection import KFold


def expand_grid(grid):
    ks=list(grid)
    return [dict(zip(ks,values)) for values in product(*(v if isinstance(v,list) else [v] for v in (grid[k] for k in ks)))]


def energy_metrics(e_true,e_fast,alpha_hat):
    residual=np.asarray(e_true)-np.asarray(e_fast)*np.asarray(alpha_hat)
    return np.array([np.sqrt(np.mean(residual**2)),np.percentile(residual,75)-np.percentile(residual,25)])


def score_columns(metrics):
    a=np.asarray(metrics,float); mn=a.min(axis=0);mx=a.max(axis=0)
    scaled=np.divide(a-mn,mx-mn,out=np.zeros_like(a),where=(mx>mn))
    return np.hypot(1-scaled[:,0],1-scaled[:,1])


def nested_select_energy(fit_predict,X,alpha,e_fast,e_true,grid,seed=0,outer_k=5,inner_k=5):
    """Re-run grid search on D70 ONLY; can be expensive for neural models."""
    if (outer_k,inner_k)!=(5,5):raise ValueError('Thesis Stages II/III require 5x5 nested CV')
    X=np.asarray(X);alpha=np.asarray(alpha).reshape(-1);e_fast=np.asarray(e_fast).reshape(-1);e_true=np.asarray(e_true).reshape(-1)
    if not all(len(a)==len(X) for a in [alpha,e_fast,e_true]):raise ValueError('input lengths do not match')
    configs=expand_grid(grid)
    if not configs:raise ValueError('empty grid')
    logs=[]
    for k,(train,val) in enumerate(KFold(n_splits=5,shuffle=True,random_state=seed).split(X),1):
        inner=list(KFold(n_splits=5,shuffle=True,random_state=seed+k).split(train))
        tri0,va0=inner[0];pilot=[]
        for conf in configs:
            yh=fit_predict(X[train][tri0],X[train][va0],alpha[train][tri0],conf)
            pilot.append(energy_metrics(e_true[train][va0],e_fast[train][va0],yh))
        order=np.argsort(-score_columns(pilot),kind='stable')
        nshort=min(len(configs),64,max(8,int(np.ceil(.2*len(configs)))))
        shortlist=[configs[int(i)] for i in order[:nshort]]
        mean_scores=[]
        for conf in shortlist:
            vals=[]
            for ti,vi in inner:
                pred=fit_predict(X[train][ti],X[train][vi],alpha[train][ti],conf)
                vals.append(energy_metrics(e_true[train][vi],e_fast[train][vi],pred))
            mean_scores.append(np.mean(vals,axis=0))
        best=shortlist[int(np.argmax(score_columns(mean_scores)))]
        pred=fit_predict(X[train],X[val],alpha[train],best)
        out=energy_metrics(e_true[val],e_fast[val],pred)
        logs.append({'fold':k,'best_params':best,'outer_rmse_native_energy_units':float(out[0]),'outer_iqr_native_energy_units':float(out[1]),'candidate_count':len(configs),'shortlist_count':len(shortlist)})
    keys=[json.dumps(item['best_params'],sort_keys=True) for item in logs]
    counts=Counter(keys);mx=max(counts.values())
    chosen=next(k for k in keys if counts[k]==mx)
    return {'selected_params':json.loads(chosen),'outer_folds':logs,'scope':'conditional nested 5x5 f_ML given g_ML fixed on D70','metric_domain':'calibrated-energy error','selection':'first inner fold / shortlist / five inner folds / modal outer configs'}
