"""Regression tests that target common methodological vulnerabilities."""
import numpy as np
import pandas as pd
import pytest
from lorenzetti_repro.representations import Representation
from lorenzetti_repro.metrics import ear_curve
from lorenzetti_repro.analysis_tables import compute_event_tables,compute_D0
from lorenzetti_repro.statistics import directional_paired_inference

def _correlated_rings(n=120,seed=32):
    g=np.random.default_rng(seed)
    X=g.normal(size=(n,100));y=X[:,0]
    X[:,1]=y+0.03*g.standard_normal(n)
    return X,y

def test_correlation_threshold_is_not_feature_count():
    X,y=_correlated_rings()
    historical=Representation('CORR-90',stage=1).fit(X,y)
    assert len(historical.indices_)==2 and set(historical.indices_)=={0,1}
    modern=Representation('CORR-10',stage=2).fit(X,y)
    assert len(modern.indices_)==10

def test_rings_two_exact_expert_choice():
    X,y=_correlated_rings()
    representation=Representation('Rings/2',stage=2).fit(X,y)
    original=[v-1 for start,end in [(1,4),(9,40),(73,76),(81,84),(89,94)] for v in range(start,end+1)]
    assert representation.indices_.tolist()==original
    assert representation.transform(X).shape==(len(X),50)

def test_ear_strict_less_than_and_grid():
    truth=np.array([5,7,9,11.]);pred=np.array([4,7,7,10.])
    x,v=ear_curve(truth,pred,delta_max=3,points=4)
    assert np.allclose(x,[0,1,2,3])
    assert np.allclose(v,[0,.25,.75,1.])

def test_d0_normalization_requires_recorded_time():
    metrics=pd.DataFrame({'configuration':['A','B'], 'rmse_MeV':[10,20], 'iqr_MeV':[5,10], 'F':[10,100]})
    with pytest.raises(ValueError):compute_D0(metrics,pd.DataFrame())
    scores=compute_D0(metrics,pd.DataFrame({'configuration':['A','B'],'time_ms_per_event':[1,2]}))
    assert scores.iloc[0]['configuration']=='A'
    assert scores.D0.iloc[0]==0.
    assert np.isclose(scores.D0.iloc[1],2.)

def test_paired_directional_statistic():
    rng=np.random.default_rng(33);n=400
    truth=np.linspace(20000,200000,n)
    fast=truth-2000+rng.normal(0,700,size=n)
    alpha=(truth-1000+rng.normal(0,300,size=n))/fast
    r=directional_paired_inference(truth,fast,alpha)
    assert r['n']==n
    assert r['mean_paired_shift_MeV']<0
    assert 0<=r['t_one_sided_p']<=1
    assert 0<=r['pitman_morgan_one_sided_p']<=1
