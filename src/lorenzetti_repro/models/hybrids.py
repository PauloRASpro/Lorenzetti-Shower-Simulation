"""Implementation of three Measurement residual/stacked models (Thesis Chapter 4).

These are thesis-aligned runnable reconstructions; not certified byte-identical
historic implementations. RFBDT follows the source notebook's RF->BDT
residual correction. Stacking uses OOB RF plus OOF BDT in Ridge training.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split


def _rf_params(p, oob=False):
    return dict(n_estimators=int(p.get('rf_n_estimators',1000)),
                max_depth=p.get('rf_max_depth',10),
                min_samples_leaf=int(p.get('rf_min_samples_leaf',24)),
                min_samples_split=int(p.get('rf_min_samples_split',24)),
                max_features=p.get('rf_max_features',1.0),
                bootstrap=bool(p.get('rf_bootstrap',True)),
                max_samples=p.get('rf_max_samples',0.7),
                ccp_alpha=float(p.get('rf_ccp_alpha',0.0)),
                n_jobs=int(p.get('rf_n_jobs',1)),
                random_state=int(p.get('random_state',0)),
                oob_score=bool(oob))


def _gb_params(p):
    return dict(n_estimators=int(p.get('gb_n_estimators',1000)),
                num_leaves=int(p.get('gb_num_leaves',127)),
                min_child_samples=int(p.get('gb_min_child_samples',24)),
                feature_fraction=float(p.get('gb_feature_fraction',1.0)),
                bagging_fraction=float(p.get('gb_bagging_fraction',0.7)),
                bagging_freq=int(p.get('gb_bagging_freq',1)),
                learning_rate=float(p.get('gb_learning_rate',0.05)),
                lambda_l2=float(p.get('gb_lambda_l2',0.0)),
                objective='regression',
                random_state=int(p.get('random_state',0)),
                n_jobs=int(p.get('gb_num_threads',1)),
                verbosity=-1)


def _fit_bdt(X, y, params, early_stop=True):
    import lightgbm as lgb
    X=np.asarray(X,dtype=np.float32)
    y=np.asarray(y,dtype=np.float64).reshape(-1)
    model=lgb.LGBMRegressor(**_gb_params(params))
    v=float(params.get('val_fraction',0.1))
    stop=int(params.get('early_stopping_rounds',50))
    if early_stop and len(y)>=20 and 0<v<0.5 and stop>0:
        a,b=train_test_split(np.arange(len(y)),test_size=v,random_state=int(params.get('random_state',0)))
        model.fit(X[a],y[a],eval_set=[(X[b],y[b])],eval_metric='l2',callbacks=[lgb.early_stopping(stop,verbose=False)])
    else: model.fit(X,y)
    return model


class ResidualHybrid:
    def __init__(self,order='RFBDT',**params):
        if order not in ('RFBDT','BDTRF'): raise ValueError('order must be RFBDT or BDTRF')
        self.order,self.params=order,params
    def fit(self,X,y):
        X=np.asarray(X,dtype=np.float32);y=np.asarray(y,dtype=np.float64).reshape(-1)
        if self.order=='RFBDT':
            self.first_=RandomForestRegressor(**_rf_params(self.params)).fit(X,y)
            residual=y-self.first_.predict(X)
            self.second_=_fit_bdt(X,residual,self.params)
        else:
            self.first_=_fit_bdt(X,y,self.params)
            residual=y-self.first_.predict(X)
            self.second_=RandomForestRegressor(**_rf_params(self.params)).fit(X,residual)
        return self
    def predict(self,X):
        X=np.asarray(X,dtype=np.float32)
        return self.first_.predict(X)+self.second_.predict(X)


class RFBDTS:
    """Ridge on RF-OOB and BDT-OOF predictions (3-fold internal stacking).

    Internal stacking folds are NOT the outer/inner nested CV folds.
    """
    def __init__(self,**params):self.params=params
    def fit(self,X,y):
        X=np.asarray(X,dtype=np.float32);y=np.asarray(y,dtype=np.float64).reshape(-1)
        if len(X)<25:raise ValueError('Need at least 25 examples for stable OOB/OOF stacking')
        rf_oob=RandomForestRegressor(**_rf_params(self.params,oob=True)).fit(X,y)
        oob=np.asarray(rf_oob.oob_prediction_,dtype=float)
        if not np.isfinite(oob).all():raise ValueError('OOB predictions nonfinite; increase n_estimators')
        k=int(self.params.get('n_folds',3))
        pred=np.full(len(y),np.nan,dtype=float)
        for tr,va in KFold(n_splits=k,shuffle=True,random_state=int(self.params.get('random_state',0))).split(X):
            bdt=_fit_bdt(X[tr],y[tr],self.params)
            pred[va]=bdt.predict(X[va])
        if not np.isfinite(pred).all():raise ValueError('Missing OOF prediction')
        self.meta_=Ridge(alpha=float(self.params.get('meta_alpha',1.0))).fit(np.column_stack([oob,pred]),y)
        self.rf_=RandomForestRegressor(**_rf_params(self.params)).fit(X,y)
        self.bdt_=_fit_bdt(X,y,self.params)
        return self
    def predict(self,X):
        X=np.asarray(X,dtype=np.float32)
        return self.meta_.predict(np.column_stack([self.rf_.predict(X),self.bdt_.predict(X)]))
