import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import AnnualReservoirDataset
from feature_engineering import FeatureEngineer
from multiscale_predictor import _compute_cascade_inflows


class FeatureEngineerCausalityTests(unittest.TestCase):
    def test_create_features_ignores_outflow_argument(self):
        engineer = FeatureEngineer()
        inflow = np.arange(72, dtype=np.float32).reshape(12, 6)
        outflow_a = np.zeros((12, 6), dtype=np.float32)
        outflow_b = np.full((12, 6), 9999.0, dtype=np.float32)

        features_a = engineer.create_features(inflow, outflow_data=outflow_a)
        features_b = engineer.create_features(inflow, outflow_data=outflow_b)

        self.assertEqual(features_a.shape, (12, 22))
        self.assertTrue(np.allclose(features_a, features_b))


class AnnualDatasetCausalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = AnnualReservoirDataset(
            data_dir='test',
            years=[2022],
            normalize=False,
            use_log_transform=False,
            fit_transforms=True,
        )
        cls.sample = cls.dataset[0]

    def test_q_in_uses_only_exogenous_inflows(self):
        q_in = self.sample['q_in'].numpy()
        head_inflow = self.sample['head_inflow'].numpy()
        interval_inflow = self.sample['interval_inflow'].numpy()

        self.assertTrue(np.allclose(q_in[:, 0], head_inflow))
        self.assertTrue(np.allclose(q_in[:, 1:], interval_inflow[:, : q_in.shape[1] - 1]))

    def test_original_outflow_is_supervision_only(self):
        q_in = self.sample['q_in'].numpy()
        original_outflow = self.sample['original_outflow'].numpy()
        self.assertFalse(np.allclose(q_in[:, 1:], original_outflow[:, :-1]))


class CascadeInflowRecurrenceTests(unittest.TestCase):
    def test_compute_cascade_inflows_uses_predicted_upstream_outflow(self):
        q_out = torch.tensor(
            [[[10.0, 20.0, 30.0],
              [11.0, 21.0, 31.0]]],
            dtype=torch.float32,
        )
        head_inflow = torch.tensor([[100.0, 101.0]], dtype=torch.float32)
        interval_inflow = torch.tensor(
            [[[1.0, 2.0],
              [3.0, 4.0]]],
            dtype=torch.float32,
        )

        q_in = _compute_cascade_inflows(q_out, head_inflow, interval_inflow)
        expected = torch.tensor(
            [[[100.0, 11.0, 22.0],
              [101.0, 14.0, 25.0]]],
            dtype=torch.float32,
        )

        self.assertTrue(torch.allclose(q_in, expected))


if __name__ == '__main__':
    unittest.main()
