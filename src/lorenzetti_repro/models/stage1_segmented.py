"""Monte Carlo-only Stage I segmented BDT using thesis Eq. (5.7)-(5.8).

The fixed published GA boundaries are reused (we do NOT claim to rerun the
historic GA optimization). Etrue (truth) routes an event to a regional model.
This route is NOT deployable in a real HLT without a separate new study.
"""
import numpy as np
from lightgbm import LGBMRegressor

ETA_BOUNDS=np.array([0.0,0.43,0.83,1.46],dtype=float)
TRUE_E_BOUNDS_MEV=np.array([0.0,44700.0,71500.0,103500.0,164300.0,499900.0],dtype=float)

def routing_bins(eta,true_energy_mev):
    e=np.asarray(true_energy_mev,dtype=float).ravel()
    a=np.abs(np.asarray(eta,dtype=float).ravel())
    if len(a)!=len(e):raise ValueError('eta/true length mismatch')
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(e)):raise ValueError('nonfinite routing variable')
    if np.any(a<ETA_BOUNDS[0]) or np.any(a>ETA_BOUNDS[-1]):raise ValueError('eta outside published Stage I segmentation')
    if np.any(e<TRUE_E_BOUNDS_MEV[0]) or np.any(e>TRUE_E_BOUNDS_MEV[-1]):raise ValueError('Etrue outside published Stage I segmentation')
    aidx=np.minimum(np.searchsorted(ETA_BOUNDS[1:],a,side='right'),2)
    eidx=np.minimum(np.searchsorted(TRUE_E_BOUNDS_MEV[1:],e,side='right'),4)
    return aidx*5+eidx

class MCSegmentedBDT:
    def __init__(self,**params):self.params=params
    def fit(self,X,y_alpha,eta,e_true_mev):
        X=np.asarray(X,dtype=float);y=np.asarray(y_alpha,dtype=float)
        groups=routing_bins(eta,e_true_mev)
        self.models_={}
        for j in range(15):
            mask=(groups==j)
            if not np.any(mask):raise ValueError(f'No training examples in published bin {j}. Must not substitute global fit silently')
            params=dict(self.params);params['verbosity']=-1
            m=LGBMRegressor(**params)
            m.fit(X[mask],y[mask])
            self.models_[j]=m
        return self
    def predict(self,X,eta,e_true_mev):
        X=np.asarray(X,dtype=float);ids=routing_bins(eta,e_true_mev)
        out=np.empty(len(X),dtype=float)
        for j,model in self.models_.items():
            mask=(ids==j)
            if np.any(mask):out[mask]=model.predict(X[mask])
        return out
