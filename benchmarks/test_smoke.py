"""Small deterministic technical checks: python -m unittest benchmarks.test_smoke."""
import unittest
import numpy as np
from .controlled_data import generate_dataset,construct_input_arrays
from .patches import PatchGeometry
from . import models,injections,metrics

class BenchmarkSmoke(unittest.TestCase):
    def test_controlled_design_and_buffer_pairing(self):
        d=generate_dataset('background_step',2026411201,'heavy_tails_t5')
        a,_=construct_input_arrays(d,0,4000);b,_=construct_input_arrays(d,0,8000)
        self.assertEqual(a['deltas'].shape,(144,140,12))
        np.testing.assert_array_equal(a['val_ids'],b['val_ids'])
        np.testing.assert_array_equal(a['deltas_raw'],b['deltas_raw'])
        self.assertFalse(set(a['train_ids']) & set(a['cal_ids']))
        self.assertFalse(set(a['train_ids']) & set(a['val_ids']))

    def test_patch_constant_preservation(self):
        xy=np.array([[0,0],[2000,0],[0,2000],[2000,2000]],float)
        p=PatchGeometry(xy).patches(np.full((4,12),.4),np.zeros(12),np.ones(12))
        np.testing.assert_allclose(p,.4,atol=1e-6)
        self.assertEqual(p.shape,(4,12,32,32))

    def test_all_comparison_models(self):
        x=np.random.default_rng(391).uniform(0,1,(8,12,32,32)).astype(np.float32)
        for method in models.METHODS:
            with self.subTest(method=method):
                model=models.fit(x,method,seed=891,epochs=1,threads=1)
                p=model.predict(x[:3]);self.assertEqual(p['reconstruction'].shape,(3,12,32,32))
                self.assertTrue(np.isfinite(p['score']).all())

    def test_regional_injections(self):
        xx,yy=np.meshgrid(np.arange(40)*2000.,np.arange(30)*2000.)
        xy=np.column_stack((xx.ravel(),yy.ravel()));ids=np.array([f'P{i:04d}' for i in range(len(xy))])
        p=injections.build_injections('cv_b4_fold1',frame=dict(xy=xy,ids=ids,targets=np.asarray(injections.TARGETS)))
        self.assertEqual(injections.validate_injections(p)['status'],'PASS')

    def test_tied_metrics(self):
        p=np.array([1,1,0,0],bool)
        self.assertEqual(metrics.binary_metrics(np.ones(4),p,~p)['auc'],.5)
        self.assertEqual(metrics.binary_metrics(np.arange(4)[::-1],p,~p)['ap'],1.)

if __name__=='__main__':unittest.main()
