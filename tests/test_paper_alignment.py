import unittest
import torch
from losses import RampLoss, TerminalWindowLoss, PowerEnergySurrogate
from models import SimplexCapacityHead


class UnitHead(torch.nn.Module):
    def v2h_all(self, volume):
        return torch.ones_like(volume) * 2

    def q2zt_all(self, flow):
        return torch.ones_like(flow)


class PaperLossTests(unittest.TestCase):
    def test_equation_24_is_squared_difference(self):
        flow = torch.tensor([[[0.], [2.], [5.]]], requires_grad=True)
        loss = RampLoss(weight=1)(flow)
        self.assertAlmostEqual(loss.item(), 6.5)
        loss.backward()
        self.assertTrue(torch.isfinite(flow.grad).all())

    def test_terminal_mse_is_not_capacity_normalized(self):
        loss = TerminalWindowLoss(window_k=2, use_level=True)
        value = loss(torch.zeros(1, 2, 1), torch.zeros(1, 2, 1),
                     torch.tensor([[3.]]), torch.tensor([[1.]]),
                     torch.full((2,), 1e8), to_level=lambda x: x,
                     q_min=torch.zeros(1, 2, 1), q_max=torch.ones(1, 2, 1) * 10)
        self.assertAlmostEqual(value.item(), 4.)

    def test_raw_generation_precedes_icr_clamp(self):
        raw = SimplexCapacityHead()(torch.tensor([[[20.], [-20.]]]),
                torch.zeros(1, 2, 1), torch.ones(1, 2, 1),
                torch.full((1, 2, 1), .75), torch.zeros(1, 1),
                torch.zeros(1, 1), torch.full((2,), 1e8))
        self.assertGreater(raw[0, 0, 0].item(), 1.)
        self.assertAlmostEqual(raw.sum().item(), 1.5, places=5)

    def test_equation_23_uses_physical_watts(self):
        score = PowerEnergySurrogate(UnitHead())(
            torch.ones(1, 2, 1) * 2, torch.ones(1, 2, 1) * 2,
            torch.ones(1, 1), torch.ones(2))
        self.assertAlmostEqual(score.item(), 1000 * 9.81 * .88 * 2, places=1)


if __name__ == '__main__':
    unittest.main()
