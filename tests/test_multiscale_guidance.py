import unittest
import warnings

import torch

from feature_engineering import build_multiscale_features


class MultiScaleGuidanceTests(unittest.TestCase):
    def test_guidance_features_do_not_raise_shape_warning(self) -> None:
        B, T, R = 1, 4, 3
        q_in = torch.tensor(
            [[[120.0, 100.0, 90.0], [125.0, 102.0, 91.0], [130.0, 103.0, 92.0], [128.0, 104.0, 93.0]]],
            dtype=torch.float32,
        )
        q_min = torch.full((B, T, R), 60.0, dtype=torch.float32)
        q_max = torch.full((B, T, R), 260.0, dtype=torch.float32)
        v_min = torch.full((T, R), 90.0, dtype=torch.float32)
        v_max = torch.full((T, R), 210.0, dtype=torch.float32)
        v0 = torch.tensor([[120.0, 125.0, 130.0]], dtype=torch.float32)
        vt = torch.tensor([[140.0, 145.0, 150.0]], dtype=torch.float32)

        inputs = {
            "q_in": q_in,
            "q_min": q_min,
            "q_max": q_max,
            "delta_t": torch.ones(T, dtype=torch.float32),
            "V0": v0,
            "V_target": vt,
            "V_min": v_min,
            "V_max": v_max,
        }
        cfg = {"multiscale": {"windows": [2], "reduces": ["mean"]}}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ms = build_multiscale_features(inputs, cfg)

        self.assertIsNotNone(ms)
        assert ms is not None
        self.assertEqual(ms.shape[:3], (B, T, R))
        self.assertGreater(ms.shape[-1], 0)
        self.assertFalse(torch.isnan(ms).any())

        guidance_warnings = [
            w
            for w in caught
            if "terminal guidance feature generation failed" in str(w.message).lower()
        ]
        self.assertEqual(len(guidance_warnings), 0)


if __name__ == "__main__":
    unittest.main()
