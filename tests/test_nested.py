import numpy as np
from sklearn.linear_model import Ridge
from lorenzetti_repro.selection import nested_select_energy

def test_nested_energy_and_modal_selection():
    rng=np.random.default_rng(9)
    X=rng.normal(size=(80,3))
    truth=30000+1000*X[:,0]
    fast=truth*0.9
    alpha=truth/fast
    def fit_predict(a,b,y,p):
        return Ridge(alpha=p['alpha']).fit(a,y).predict(b)
    result=nested_select_energy(fit_predict,X,alpha,fast,truth,{'alpha':[.01,.1,1.]})
    assert len(result['outer_folds'])==5
    assert result['selected_params']['alpha'] in [.01,.1,1.]
    assert result['metric_domain']=='calibrated-energy error'
