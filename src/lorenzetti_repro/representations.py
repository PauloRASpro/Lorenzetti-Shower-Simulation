"""Thesis-aligned reconstructions fitted only on D70.
Supervised PCA component ranking follows the written thesis; this may differ
from intermediate notebook's first-n PCA function. Results must be revalidated.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,FastICA
from sklearn.feature_selection import mutual_info_regression

def _safe_corr(x,y):
    sx=np.std(x);sy=np.std(y)
    if sx<=1e-14 or sy<=1e-14:return 0.0
    v=np.corrcoef(x,y)[0,1]
    return float(v) if np.isfinite(v) else 0.0

class Representation:
    def __init__(self,name,seed=0,stage=None):
        self.name=name.upper().replace('_','-')
        self.seed=int(seed)
        self.stage=int(stage) if stage is not None else None
        self.fitted_=False
    def fit(self,rings_d70,true_d70):
        X=np.asarray(rings_d70,dtype=float)
        y=np.asarray(true_d70,dtype=float)
        if X.ndim!=2 or X.shape[1]!=100: raise ValueError('Expected exactly 100 Ringer columns')
        if len(X)!=len(y): raise ValueError('X/y mismatch')
        if self.name in ('RINGS100','RINGS[1:100]','FULL'):
            self.indices_=np.arange(100);self.mode_='identity'
        elif self.name=='RINGS/2':
            # Fixed 1-based sampling specified in thesis §5.5 (F=50).
            blocks=[range(1,5),range(9,41),range(73,77),range(81,85),range(89,95)]
            self.indices_=np.array([v-1 for block in blocks for v in block],dtype=int)
            assert len(self.indices_)==50 and len(np.unique(self.indices_))==50
            self.mode_='select'
        elif '-THR-' in self.name or (self.stage==1 and self.name.split('-')[0] in ('CORR','MUTI')):

            if '-THR-' in self.name:kind,pct=self.name.split('-THR-',1)
            else:kind,pct=self.name.split('-',1)
            thr=float(pct)/100.0
            if kind not in ('CORR','MUTI'):raise ValueError('Thresholds only for CORR/MUTI')
            if kind=='CORR':scores=np.array([abs(_safe_corr(X[:,i],y)) for i in range(100)])
            else:scores=mutual_info_regression(X,y,random_state=self.seed)
            self.indices_=np.flatnonzero(scores>=thr)
            if not len(self.indices_):raise ValueError(f'{self.name} selected zero rings with this estimator; no silent fallback')
            self.mode_='select'
        else:
            kind, n_text=self.name.split('-',1)
            n=int(n_text)
            if not (1<=n<=100):raise ValueError('n outside 1..100')
            if kind in ('CORR','MUTI'):
                if kind=='CORR':
                    scores=np.array([abs(_safe_corr(X[:,i],y)) for i in range(100)])
                else:
                    scores=mutual_info_regression(X,y,random_state=self.seed)
                self.indices_=np.argsort(-scores,kind='stable')[:n]
                self.mode_='select'
            elif kind in ('PCA','ICA'):
                self.scale_=StandardScaler().fit(X)
                Xz=self.scale_.transform(X)
                if kind=='PCA':
                    # Thesis: fit 100 components, then rank supervised relevance to Etrue.
                    self.transformer_=PCA(n_components=100,random_state=self.seed).fit(Xz)
                    Z=self.transformer_.transform(Xz)
                    self.indices_=np.argsort(-np.array([abs(_safe_corr(Z[:,i],y)) for i in range(100)]),kind='stable')[:n]
                else:
                    self.transformer_=FastICA(n_components=100,random_state=self.seed,whiten='unit-variance',max_iter=2000,tol=1e-4).fit(Xz)
                    Z=self.transformer_.transform(Xz)
                    self.indices_=np.argsort(-np.array([abs(_safe_corr(Z[:,i],y)) for i in range(100)]),kind='stable')[:n]
                self.mode_='transform'
            else: raise ValueError(f'Unsupported {self.name}')
        self.fitted_=True
        return self
    def transform(self,rings):
        if not self.fitted_:raise RuntimeError('Representation must be fitted on D70 before D30.')
        X=np.asarray(rings,dtype=float)
        if X.ndim!=2 or X.shape[1]!=100:raise ValueError('Expected 100 rings')
        if self.mode_=='identity':return X
        if self.mode_=='select':return X[:,self.indices_]
        return self.transformer_.transform(self.scale_.transform(X))[:,self.indices_]
    def report(self):
        return {'name':self.name,'indices':self.indices_.tolist(),'dimensionality':int(len(self.indices_)),
                'fit_scope':'D70 only','stage':self.stage}
