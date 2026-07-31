import unittest

import torch

from constraint.hierarchical_projection import HierarchicalProjection
from losses import CompositeSchedulingLoss, TerminalWindowLoss
from multiscale_predictor import _compute_terminal_reachability


class TerminalReachabilityTests(unittest.TestCase):
    def test_reachable_terminal_target(self) -> None:
        q_min = torch.zeros((1, 2, 1), dtype=torch.float32)
        q_max = torch.full((1, 2, 1), 10.0, dtype=torch.float32)
        head_inflow = torch.tensor([[10.0, 10.0]], dtype=torch.float32)
        V0 = torch.tensor([[0.0]], dtype=torch.float32)
        dt = torch.tensor([1.0e8, 1.0e8], dtype=torch.float32)
        V_min = torch.zeros((1, 2, 1), dtype=torch.float32)
        V_max = torch.full((1, 2, 1), 100.0, dtype=torch.float32)
        V_target = torch.tensor([[15.0]], dtype=torch.float32)

        result = _compute_terminal_reachability(
            q_min=q_min,
            q_max=q_max,
            head_inflow=head_inflow,
            interval_inflow=None,
            V0=V0,
            delta_t=dt,
            V_min=V_min,
            V_max=V_max,
            V_target=V_target,
            reservoir_names=["A"],
        )

        report = result["report"]["per_reservoir"]["A"]
        self.assertTrue(report["reachable"])
        self.assertEqual(report["status"], "reachable")
        self.assertAlmostEqual(report["min_reachable_terminal"], 0.0, places=5)
        self.assertAlmostEqual(report["max_reachable_terminal"], 20.0, places=5)

    def test_unreachable_when_target_above_joint_upper_bound(self) -> None:
        q_min = torch.zeros((1, 2, 1), dtype=torch.float32)
        q_max = torch.full((1, 2, 1), 10.0, dtype=torch.float32)
        head_inflow = torch.tensor([[10.0, 10.0]], dtype=torch.float32)
        V0 = torch.tensor([[0.0]], dtype=torch.float32)
        dt = torch.tensor([1.0e8, 1.0e8], dtype=torch.float32)
        V_min = torch.zeros((1, 2, 1), dtype=torch.float32)
        V_max = torch.full((1, 2, 1), 100.0, dtype=torch.float32)
        V_target = torch.tensor([[25.0]], dtype=torch.float32)

        result = _compute_terminal_reachability(
            q_min=q_min,
            q_max=q_max,
            head_inflow=head_inflow,
            interval_inflow=None,
            V0=V0,
            delta_t=dt,
            V_min=V_min,
            V_max=V_max,
            V_target=V_target,
            reservoir_names=["A"],
        )

        report = result["report"]["per_reservoir"]["A"]
        self.assertFalse(report["reachable"])
        self.assertEqual(report["status"], "insufficient_upper_reach")
        self.assertAlmostEqual(report["gap_to_target"], 5.0, places=5)

    def test_unreachable_when_target_below_joint_lower_bound(self) -> None:
        q_min = torch.zeros((1, 2, 1), dtype=torch.float32)
        q_max = torch.full((1, 2, 1), 3.0, dtype=torch.float32)
        head_inflow = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        V0 = torch.tensor([[20.0]], dtype=torch.float32)
        dt = torch.tensor([1.0e8, 1.0e8], dtype=torch.float32)
        V_min = torch.zeros((1, 2, 1), dtype=torch.float32)
        V_max = torch.full((1, 2, 1), 100.0, dtype=torch.float32)
        V_target = torch.tensor([[5.0]], dtype=torch.float32)

        result = _compute_terminal_reachability(
            q_min=q_min,
            q_max=q_max,
            head_inflow=head_inflow,
            interval_inflow=None,
            V0=V0,
            delta_t=dt,
            V_min=V_min,
            V_max=V_max,
            V_target=V_target,
            reservoir_names=["A"],
        )

        report = result["report"]["per_reservoir"]["A"]
        self.assertFalse(report["reachable"])
        self.assertEqual(report["status"], "excess_lower_reach")
        self.assertAlmostEqual(report["min_reachable_terminal"], 14.0, places=5)
        self.assertAlmostEqual(report["gap_to_target"], 9.0, places=5)

    def test_unreachable_terminal_uses_best_effort_target(self) -> None:
        proj = HierarchicalProjection(max_iters=4, post_clamp_rebalance_iters=4, post_clamp_tol=1e-6)
        q_raw = torch.zeros((1, 2, 1), dtype=torch.float32)
        q_in = torch.tensor([[[10.0], [10.0]]], dtype=torch.float32)
        V0 = torch.tensor([[0.0]], dtype=torch.float32)
        V_target = torch.tensor([[25.0]], dtype=torch.float32)
        q_min = torch.zeros((1, 2, 1), dtype=torch.float32)
        q_max = torch.full((1, 2, 1), 10.0, dtype=torch.float32)
        V_min = torch.zeros((1, 2, 1), dtype=torch.float32)
        V_max = torch.full((1, 2, 1), 100.0, dtype=torch.float32)
        dt = torch.tensor([1.0e8, 1.0e8], dtype=torch.float32)

        q_proj, V_proj, stats = proj(
            q_raw=q_raw,
            q_in=q_in,
            V0=V0,
            V_target=V_target,
            q_min=q_min,
            q_max=q_max,
            Vmin=V_min,
            Vmax=V_max,
            delta_t=dt,
            terminal_reachable_mask=torch.tensor([[False]]),
            terminal_best_effort_target=torch.tensor([[20.0]], dtype=torch.float32),
        )

        self.assertAlmostEqual(float(V_proj[0, -1, 0].item()), 20.0, places=4)
        self.assertFalse(bool(stats["infeasible"][0, 0].item()))
        self.assertAlmostEqual(float(stats["requested_residual"][0, 0].item()), 5.0, places=4)
        self.assertAlmostEqual(float(stats["terminal_effective_target"][0, 0].item()), 20.0, places=4)

    def test_reachable_terminal_target_is_hard_enforced(self) -> None:
        proj = HierarchicalProjection(max_iters=4, post_clamp_rebalance_iters=4, post_clamp_tol=1e-6)
        q_raw = torch.zeros((1, 2, 1), dtype=torch.float32)
        q_in = torch.tensor([[[10.0], [10.0]]], dtype=torch.float32)
        V0 = torch.tensor([[0.0]], dtype=torch.float32)
        V_target = torch.tensor([[15.0]], dtype=torch.float32)
        q_min = torch.zeros((1, 2, 1), dtype=torch.float32)
        q_max = torch.full((1, 2, 1), 10.0, dtype=torch.float32)
        V_min = torch.zeros((1, 2, 1), dtype=torch.float32)
        V_max = torch.full((1, 2, 1), 100.0, dtype=torch.float32)
        dt = torch.tensor([1.0e8, 1.0e8], dtype=torch.float32)

        _, V_proj, stats = proj(
            q_raw=q_raw,
            q_in=q_in,
            V0=V0,
            V_target=V_target,
            q_min=q_min,
            q_max=q_max,
            Vmin=V_min,
            Vmax=V_max,
            delta_t=dt,
            terminal_reachable_mask=torch.tensor([[True]]),
            terminal_best_effort_target=torch.tensor([[20.0]], dtype=torch.float32),
        )

        self.assertAlmostEqual(float(V_proj[0, -1, 0].item()), 15.0, places=4)
        self.assertAlmostEqual(float(stats["requested_residual"][0, 0].item()), 0.0, places=4)
        self.assertAlmostEqual(float(stats["terminal_effective_target"][0, 0].item()), 15.0, places=4)

    def test_loss_uses_effective_terminal_target(self) -> None:
        loss_fn = CompositeSchedulingLoss(
            terminal_loss=TerminalWindowLoss(window_k=2, use_level=False),
            w_flow=0.0,
            w_term=1.0,
        )
        q_phys = torch.zeros((1, 2, 1), dtype=torch.float32)
        q_in = torch.tensor([[[10.0], [10.0]]], dtype=torch.float32)
        V0 = torch.tensor([[0.0]], dtype=torch.float32)
        dt = torch.tensor([1.0e8, 1.0e8], dtype=torch.float32)
        targets = torch.zeros_like(q_phys)

        loss = loss_fn(
            q_phys,
            targets,
            physical_predictions=q_phys,
            q_in=q_in,
            V0=V0,
            V_target=torch.tensor([[25.0]], dtype=torch.float32),
            delta_t_steps=dt,
            terminal_effective_target=torch.tensor([[20.0]], dtype=torch.float32),
        )

        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
