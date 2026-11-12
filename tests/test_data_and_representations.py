import unittest
import numpy as np
import pandas as pd
from lorenzetti_repro.dataset import split_70_30,EXPECTED_EVENTS,RINGS
from lorenzetti_repro.representations import Representation
class DataTest(unittest.TestCase):
    def test_split(self):
        d=pd.DataFrame({'x':range(100)})
        a,b=split_70_30(d)
        self.assertEqual((len(a),len(b)),(70,30))
        self.assertEqual(len(set(a).intersection(b)),0)
    def test_repr_only_fitted_on_dev(self):
        rng=np.random.default_rng(3);X=rng.normal(size=(80,100));y=X[:,6]*5
        g=Representation('CORR-10').fit(X[:55],y[:55]);z=g.transform(X[55:]);self.assertEqual(z.shape,(25,10))
        self.assertIn(6,g.indices_)
if __name__=='__main__':unittest.main()
