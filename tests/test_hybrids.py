import numpy as np
from lorenzetti_repro.models.hybrids import ResidualHybrid,RFBDTS
from lorenzetti_repro.models.stage1_segmented import routing_bins
from lorenzetti_repro.representations import Representation

def _dataset():
    rng=np.random.default_rng(1)
    x=rng.normal(size=(140,5)).astype('float32')
    y=2+.2*x[:,0]-.3*x[:,2]+rng.normal(size=140)*.08
    return x,y

def test_rfbdt_predictions_finite():
    X,y=_dataset()
    p={'rf_n_estimators':25,'rf_min_samples_leaf':2,'rf_min_samples_split':3,'rf_max_depth':5,
       'gb_n_estimators':35,'gb_num_leaves':12,'gb_min_child_samples':5,'early_stopping_rounds':8,
       'random_state':0,'rf_n_jobs':1,'gb_num_threads':1}
    m=ResidualHybrid(order='RFBDT',**p).fit(X,y)
    assert np.isfinite(m.predict(X[:10])).all()

def test_stacking_oob_oof():
    X,y=_dataset()
    p={'rf_n_estimators':45,'rf_min_samples_leaf':2,'rf_min_samples_split':3,'rf_max_depth':5,
       'gb_n_estimators':35,'gb_num_leaves':12,'gb_min_child_samples':5,'early_stopping_rounds':8,
       'n_folds':3,'random_state':0,'rf_n_jobs':1,'gb_num_threads':1}
    m=RFBDTS(**p).fit(X,y)
    assert m.meta_.coef_.shape==(2,)
    assert np.isfinite(m.predict(X[:10])).all()

def test_stage1_truth_binning():
    from lorenzetti_repro.models.stage1_segmented import routing_bins
    assert routing_bins([0.2,0.5,1.2],[30000,100000,300000]).tolist()==[0,7,14]

def test_stage1_threshold_not_cardinality():
    rng=np.random.default_rng(3);X=rng.normal(size=(230,100)); y=20*X[:,8]+15*X[:,74]
    g=Representation('CORR_THR_50').fit(X,y)
    assert 8 in g.indices_ and 74 in g.indices_
    assert len(g.indices_)!=50
