"""Executable training and event-level artifact writer for three research stages.

Uses the shared D70/D30 split, fitted representation on D70, and the stage
specific selection protocol. Serializes the experiment DataFrames with the
same names/columns used by the source research notebook.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from .dataset import RINGS,TRUE,FAST,ETA,alpha_target,read_dataset,split_70_30,EXPECTED_EVENTS
from .representations import Representation
from .selection import nested_select_energy
from .model_runner import fit_predict_model
from .models.hybrids import ResidualHybrid, RFBDTS

# Each model is actually fitted, not substituted with another family.
ARTICLE_MODELS={'tns':('mlr','preg','dtree','bdt','bspl'),
                'measurement':('dtree','bdt','rf','bdtrf','rfbdt','rfbdts'),
                'information_fusion':('bdt','rfbdt','dnn','gnn')}

REPRESENTATIONS={'tns':('rings100','CORR-50','CORR-70','CORR-80','CORR-90','MUTI-50','MUTI-70','MUTI-80','MUTI-90','PCA-2','PCA-5','PCA-10','PCA-100','ICA-2','ICA-5','ICA-10','ICA-100'),
 'measurement':('rings100','Rings/2','CORR-2','CORR-5','CORR-10','CORR-50','MUTI-2','MUTI-5','MUTI-10','MUTI-50','PCA-2','PCA-5','PCA-10','PCA-100','ICA-2','ICA-5','ICA-10','ICA-100'),
 'information_fusion':('rings100','Rings/2','CORR-2','CORR-5','CORR-10','CORR-50','MUTI-2','MUTI-5','MUTI-10','MUTI-50','PCA-2','PCA-5','PCA-10','PCA-100','ICA-2','ICA-5','ICA-10','ICA-100')}
STAGE={'tns':1,'measurement':2,'information_fusion':3}


def fit_model_with_object(name,Xtr,Xte,ytr,p):
    """Return fitted object, predicted calibration factor, fit/inference timings."""
    p=dict(p);start=time.perf_counter()
    if name=='mlr':model=LinearRegression(**p)
    elif name=='preg':model=make_pipeline(PolynomialFeatures(degree=p.pop('degree',2),include_bias=False),LinearRegression(**p))
    elif name=='dtree':model=DecisionTreeRegressor(**p)
    elif name=='bspl':
        from .models.bspl_original_cell513 import b_splines_regression
        model,pred,_,_,_,tp=b_splines_regression(Xtr,Xte,ytr,None,**p)
        return model,np.asarray(pred),float(getattr(model,'t_train',0)),float(tp)
    elif name=='rf':model=RandomForestRegressor(**p)
    elif name=='bdt':
        from lightgbm import LGBMRegressor
        model=LGBMRegressor(**p,verbosity=-1)
    elif name in ('bdtrf','rfbdt'):model=ResidualHybrid(name.upper(),**p)
    elif name=='rfbdts':model=RFBDTS(**p)
    elif name=='dnn':
        from .models.dnn_original_cell527 import dnn_regression
        model,pred,_,_,_,tp=dnn_regression(Xtr,Xte,ytr,None,**p)
        return model,np.asarray(pred),float(getattr(model,'t_train',0)),float(tp)
    elif name=='gnn':
        from .models.gnn_original_cell562 import gnn_pure_regression
        model,pred,_,_,_,tp=gnn_pure_regression(Xtr,Xte,ytr,None,**p)
        return model,np.asarray(pred),float(getattr(model,'t_train',time.perf_counter()-start)),float(tp)
    else:raise ValueError(name)
    model.fit(Xtr,ytr);tfit=time.perf_counter()-start
    start=time.perf_counter();pred=np.asarray(model.predict(Xte),dtype=float);tp=time.perf_counter()-start
    return model,pred.ravel(),tfit,tp


def label(name,rep):
    name={'mlr':'MLR','preg':'PREG','dtree':'DTREE','bdt':'BDT','bspl':'BSPL',
          'rf':'RFOR','rfbdt':'RFBDT','bdtrf':'BDTRF','rfbdts':'RFBDTS','dnn':'DNN','gnn':'GNN'}[name]
    key='Rings[1:100]' if rep=='rings100' else rep
    return name+' no-Bin: '+key


def read_training_params(root,article,name,rep,mode):
    """Use the candidate grid for model selection; never guess missing results."""
    stage=STAGE[article]
    # Reference baselines in Stage III retain their tree-study configurations.
    config=Path(root)/'configs'/f'stage{2 if article=="information_fusion" and name in ("bdt","rfbdt") else stage}'
    if stage==1:
        d=json.loads((config/'documented_candidate_configurations.json').read_text(encoding='utf-8'))
        if name=='bspl':
            return {'degree':3,'n_knots':9,'knots':'quantile','extrapolation':'linear',
                    'include_bias':False,'sparse_output':True,'estimator':'ridge',
                    'alpha':1e-3,'ridge_solver':'sparse_cg','max_iter':10000,
                    'tol':1e-4,'random_state':0},None
        if name not in d['models']:
            raise ValueError(f'No documented candidate configuration: {article}, {name}')
        return d['models'][name],None
    if mode=='nested':
        p=config/f'{name}_search.json'
        if not p.is_file():raise ValueError(f'No documented search grid in {p}')
        return None,json.loads(p.read_text(encoding='utf-8'))['grid']
    # Exact final configuration: reject undocumented combinations, not a fabricated default.
    p=config/f'{name}_{rep.lower()}.json'
    if not p.is_file():raise ValueError(f'No archived final configuration in {p}. Use mode="nested".')
    return json.loads(p.read_text(encoding='utf-8'))['parameters'],None


def train_article(article,root,dataset_csv,*,selection='nested',chosen=None,
                  save_fitted_models=True,allow_subset=False,seed=0,output_dir=None):
    """Train chosen configurations and save pickle tables in source-notebook format.

    `chosen` is a list of (model,representation), or None for all documented
    candidate representations for the selected research stage.
    """
    root=Path(root).resolve();df=read_dataset(dataset_csv)
    if not allow_subset and len(df)!=EXPECTED_EVENTS:
        raise ValueError(f'{len(df)} events instead of the {EXPECTED_EVENTS} used in the thesis')
    dev,test=split_70_30(df,random_state=seed)
    ring=df[RINGS].to_numpy(float)
    truth=df[TRUE].to_numpy(float);fast=df[FAST].to_numpy(float)
    eta=df[ETA].to_numpy(float) if ETA in df else np.repeat(np.nan,len(df))
    alpha=alpha_target(df)
    if chosen is None:
        chosen=[(m,r) for m in ARTICLE_MODELS[article] for r in REPRESENTATIONS[article]
                if m not in ('gnn','bdt','rfbdt') or r=='rings100']
    grouped={}
    if output_dir is None:
        raise ValueError('Choose a new run directory under artifacts/new_runs/<article>/<run-id>/pickles; historical artifacts/pickles is read-only.')
    output=Path(output_dir).resolve()
    originals=(root/'artifacts'/'pickles').resolve()
    if output == originals or originals in output.parents:
        raise ValueError('Training cannot write inside artifacts/pickles (historical archive).')
    new_runs=(root/'artifacts'/'new_runs').resolve()
    if new_runs not in output.parents or output.name != 'pickles':
        raise ValueError('Training output must be artifacts/new_runs/<article>/<unique-id>/pickles.')
    # Prevent silent continuation of an earlier run or accidental overwrite.
    output.mkdir(parents=True,exist_ok=False)
    fitted_dir=output.parent/'trained_models';fitted_dir.mkdir(parents=True,exist_ok=True)
    manifest=[]
    def checkpoint(method, part):
        vals={'noSEG_Full_DF':pd.concat(part['event'],ignore_index=True),
              'noSEG_erros_data':pd.DataFrame(part['error']),
              'noSEG_erros_data_alpha':pd.DataFrame(part['alpha_error']),
              'noSEG_cv_metrics_df':pd.DataFrame(part['cv']),
              'noSEG_times':pd.DataFrame(part['time']),
              'noSEG_performance_metrics':pd.DataFrame(part['score'])}
        from .original_segmentation import build_segmented_tables
        import contextlib, io
        try:
            # These are the author's original source functions. They can be
            # verbose for many bins; their print output is not scientific data.
            with contextlib.redirect_stdout(io.StringIO()):
                seg, integrated = build_segmented_tables(vals['noSEG_Full_DF'])
            vals['result_noSEG_Full_DF']=seg
            vals['integrated_result_noSEG_Full_DF']=integrated
        except (ValueError,TypeError,IndexError,ZeroDivisionError) as exc:
            raise ValueError(f'Segmentation archive failed for {method}: {exc}') from exc
        for suffix,tab in vals.items():
            if suffix.startswith('result_') or suffix.startswith('integrated_result_'):
                name=f'{suffix.split("noSEG")[0]}{method}_noSEG_Full_DF.pkl.gz'
            else:
                name=f'{method}_{suffix}.pkl.gz'
            target=output/name
            tmp=target.with_name(target.name+'.tmp')
            tab.to_pickle(tmp,compression='gzip',protocol=5)
            if target.exists():
                # Checkpoints update the aggregate within THIS new run only; historic artifacts are guarded above.
                target.unlink()
            tmp.replace(target)
    for name,rep in chosen:
        if name not in ARTICLE_MODELS[article]:raise ValueError(f'Model {name} outside {article}')
        if rep not in REPRESENTATIONS[article]:raise ValueError(f'Representation {rep} outside {article}')
        if name=='gnn' and rep!='rings100':raise ValueError('Published GNN uses all rings')
        model_name=label(name,rep)
        print('FITTING:',article,model_name,flush=True)
        start=time.perf_counter()
        g=Representation(rep,seed=0,stage=STAGE[article]).fit(ring[dev],truth[dev])
        xtr=g.transform(ring[dev]);xte=g.transform(ring[test]);t_g=time.perf_counter()-start
        params,search=read_training_params(root,article,name,rep,selection)
        cv=None
        if search is not None:
            cv=nested_select_energy(lambda a,b,c,p:fit_predict_model(name,a,b,c,p),
                          xtr,alpha[dev],fast[dev],truth[dev],search)
            params=cv['selected_params']
        if params is None:raise ValueError(f'No verified configuration for {model_name}')
        fitted,ypred,fit_s,pred_s=fit_model_with_object(name,xtr,xte,alpha[dev],params)
        if not np.isfinite(ypred).all():raise ValueError(f'Nonfinite prediction {model_name}')
        name_upper=model_name.split()[0];stash=grouped.setdefault(name_upper,{'event':[],'time':[],'score':[],'cv':[],'error':[],'alpha_error':[]})
        t=truth[test];f=fast[test];cal=ypred*f
        data=pd.DataFrame({'method':[name_upper+' no-Bin']*len(test),
            'method-selection':[model_name]*len(test), 'useRings':[rep]*len(test),
            'ypred':ypred,'yte':alpha[test], 'cluster_truthet':f,
            'mc_match_et':t,'cluster_trutheta':eta[test], 'calib_et':cal})
        stash['event'].append(data)
        residual=t-cal;residual_f=t-f
        rmse=lambda a:float(np.sqrt(np.mean(np.asarray(a)**2)))
        iqr=lambda a:float(np.quantile(a,.75)-np.quantile(a,.25))
        stash['score'].append({'Method-Selection':model_name,'RMSE (MeV)':rmse(residual),
            'IQR (MeV)':iqr(residual),'d0 (MeV)':float(np.hypot(rmse(residual),iqr(residual))),
            'RMSE (MeV) Standard FastCalo':rmse(residual_f),'IQR (MeV) Standard FastCalo':iqr(residual_f)})
        stash['time'].append({'Method-Selection':model_name,
              'Total Training-time (s)':(t_g+fit_s)/len(dev),
              'Total Prediction-time (s)':pred_s/len(test),
              'Features':int(xtr.shape[1])})
        stash['error'].append({'data_labels':model_name,'data_erros':residual.tolist()})
        stash['alpha_error'].append({'data_labels_alpha':model_name,'data_erros_alpha':(alpha[test]-ypred).tolist()})
        if cv:stash['cv'].append({'Method-Selection':model_name,
                                 'selected_params':json.dumps(params,default=str),
                                 'outer_cv':json.dumps(cv,default=str)})
        if save_fitted_models:
            path=fitted_dir/(name_upper+'_'+rep+'.joblib')
            if name=='gnn':
                import torch
                path=path.with_suffix('.pt')
                torch.save({'model_state_dict':fitted.state_dict(),'params':params,'representation':rep},path)
            else:
                import joblib
                joblib.dump({'estimator':fitted,'transformer':g,'parameters':params},path,compress=3)
        manifest.append({'model':model_name,'events':len(test),'training_s':fit_s,'predict_s':pred_s,
                         'parameters':params,'split_seed':seed,'split_D70':len(dev),'split_D30':len(test),
                         'target':'Etrue/Efast','feature_count':int(xtr.shape[1])})
        checkpoint(name_upper,stash)
    (output.parent/'manifests').mkdir(parents=True,exist_ok=True)
    (output.parent/'manifests'/f'training_{article}.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf8')
    print(f'Saved {len(manifest)} model configurations to {output}',flush=True)
    return manifest
