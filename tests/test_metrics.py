import unittest
import numpy as np
from lorenzetti_repro.metrics import assess,shared_ear,delta_lambda_90

class MetricsTest(unittest.TestCase):
    def test_perfect_calibration(self):
        truth=np.array([10.,20.,30.,40.,50.]);fast=truth*.9
        d=assess(truth,fast,truth/fast)
        self.assertAlmostEqual(d['rmse'],0.0,places=12)
        self.assertAlmostEqual(d['iqr'],0.0,places=12)
    def test_ear_non_decreasing(self):
        a=np.arange(1,101.);b=a+np.random.default_rng(0).normal(0,0.1,len(a))
        x,c,f=shared_ear(a,b,a+2)
        self.assertEqual(len(x),1000)
        self.assertTrue((np.diff(c)>=-1e-15).all())
        self.assertGreaterEqual(delta_lambda_90(x,c),0)
if __name__=='__main__':unittest.main()
