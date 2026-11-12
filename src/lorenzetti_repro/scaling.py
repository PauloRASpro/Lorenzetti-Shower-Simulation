import numpy as np
from sklearn.preprocessing import StandardScaler

def z_norm_input(X_train, X_test):
    """Fit on training only; interface of original notebook cell #258."""
    a=np.asarray(X_train,dtype=float)
    b=np.asarray(X_test,dtype=float)
    scale=StandardScaler().fit(a)
    xtr=scale.transform(a)
    xte=scale.transform(b)
    stats={'mean':scale.mean_, 'std':scale.scale_, 'columns':list(getattr(X_train,'columns',[])), 'is_df':False}
    return xtr,xte,stats
