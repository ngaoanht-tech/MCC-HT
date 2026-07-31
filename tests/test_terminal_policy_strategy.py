import unittest

import torch

from multiscale_predictor import _update_post_projection_terminal_policy


class TerminalPolicyStrategyTests(unittest.TestCase):
    def test_keep_hard_target_when_fallback_disabled(self) -> None:
        req = torch.tensor([[0.2, 0.0]], dtype=torch.float32)
        vt = torch.tensor([[10.0, 20.0]], dtype=torch.float32)
        initial_mask = torch.tensor([[True, True]])
        current_mask = torch.tensor([[True, True]])
        best_eff = torch.tensor([[11.0, 21.0]], dtype=torch.float32)

        out = _update_post_projection_terminal_policy(
            req_residual=req,
            V_terminal=vt,
            initial_reachable_mask=initial_mask,
            current_reachable_mask=current_mask,
            current_best_effort_target=best_eff,
            reservoir_names=["A", "B"],
            tol=1e-4,
            allow_fallback=False,
        )

        self.assertTrue(torch.equal(out["reachable_mask"], current_mask))
        self.assertTrue(torch.allclose(out["best_effort_target"], best_eff))
        self.assertIn("A", out["hard_gap"])
        self.assertEqual(out["fallback_gap"], {})

    def test_downgrade_to_best_effort_when_fallback_enabled(self) -> None:
        req = torch.tensor([[0.2, 0.0]], dtype=torch.float32)
        vt = torch.tensor([[9.5, 20.0]], dtype=torch.float32)
        initial_mask = torch.tensor([[True, True]])
        current_mask = torch.tensor([[True, True]])
        best_eff = torch.tensor([[11.0, 21.0]], dtype=torch.float32)

        out = _update_post_projection_terminal_policy(
            req_residual=req,
            V_terminal=vt,
            initial_reachable_mask=initial_mask,
            current_reachable_mask=current_mask,
            current_best_effort_target=best_eff,
            reservoir_names=["A", "B"],
            tol=1e-4,
            allow_fallback=True,
        )

        self.assertFalse(bool(out["reachable_mask"][0, 0].item()))
        self.assertAlmostEqual(float(out["best_effort_target"][0, 0].item()), 9.5, places=6)
        self.assertIn("A", out["hard_gap"])
        self.assertIn("A", out["fallback_gap"])

    def test_only_precheck_reachable_reservoirs_are_considered(self) -> None:
        req = torch.tensor([[0.3, 0.4]], dtype=torch.float32)
        vt = torch.tensor([[10.0, 20.0]], dtype=torch.float32)
        initial_mask = torch.tensor([[False, True]])
        current_mask = torch.tensor([[False, True]])
        best_eff = torch.tensor([[10.5, 20.5]], dtype=torch.float32)

        out = _update_post_projection_terminal_policy(
            req_residual=req,
            V_terminal=vt,
            initial_reachable_mask=initial_mask,
            current_reachable_mask=current_mask,
            current_best_effort_target=best_eff,
            reservoir_names=["A", "B"],
            tol=1e-4,
            allow_fallback=False,
        )

        self.assertNotIn("A", out["hard_gap"])
        self.assertIn("B", out["hard_gap"])


if __name__ == "__main__":
    unittest.main()
