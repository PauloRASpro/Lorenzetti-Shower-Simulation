"""Callable prediction adapters for thesis-aligned reruns; no access to D30 in fit."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def fit_predict_model(model_name,Xtr,Xva,ytr,params):
    """Return predicted alpha on Xva. Caller owns the train/test separation."""
    name=model_name.lower();p=dict(params)
    if name=='mlr':model=LinearRegression(**p)
    elif name=='preg':
        degree=int(p.pop('degree',2))
        model=make_pipeline(PolynomialFeatures(degree=degree,include_bias=False),LinearRegression(**p))
    elif name=='dtree':model=DecisionTreeRegressor(**p)
    elif name=='bspl':
        from lorenzetti_repro.models.bspl_original_cell513 import b_splines_regression
        return np.asarray(b_splines_regression(Xtr,Xva,ytr,None,**p)[1]).ravel()
    elif name=='rf':model=RandomForestRegressor(**p)
    elif name=='bdt':
        from lightgbm import LGBMRegressor
        model=LGBMRegressor(**p,verbosity=-1)
    elif name in ('rfbdt','bdtrf','rfbdts'):
        from lorenzetti_repro.models.hybrids import ResidualHybrid,RFBDTS
        model=RFBDTS(**p) if name=='rfbdts' else ResidualHybrid(name.upper(),**p)
    elif name=='dnn':
        from lorenzetti_repro.models.dnn_original_cell527 import dnn_regression
        return np.asarray(dnn_regression(Xtr,Xva,ytr,None,**p)[1]).ravel()
    elif name=='gnn':
        from lorenzetti_repro.models.gnn_original_cell562 import gnn_pure_regression
        return np.asarray(gnn_pure_regression(Xtr,Xva,ytr,None,**p)[1]).ravel()
    else:raise ValueError(f'Unknown model: {model_name}')
    model.fit(Xtr,ytr)
    return np.asarray(model.predict(Xva)).ravel()
