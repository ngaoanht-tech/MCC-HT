import unittest
import torch
from paper_physics import operating_checks, cascade_state


class IdentityCurves:
    def v2h_all(self, x): return x
    def q2zt_all(self, x): return torch.zeros_like(x)


class PhysicalChecksTests(unittest.TestCase):
    def test_real_curve_tables_accept_double_precision_checks(self):
        from pathlib import Path
        from losses import TorchCurves
        root = Path(__file__).resolve().parents[1]
        names = ['乌东德','白鹤滩','溪洛渡','向家坝','三峡','葛洲坝']
        curves = TorchCurves(str(root/'shuxing'/'curves'), str(root/'shuxing'), names)
        q = torch.ones(1,36,6)*1000
        initial = torch.tensor([[58.225,188.978,115.072,49.2905,387.949,16.185]])
        r = operating_checks(q, torch.ones(1,36)*1000, torch.zeros(1,36,5),
            initial, initial, torch.ones(36)*864000, torch.zeros_like(q),
            torch.ones_like(q)*2000, torch.zeros(36,6), torch.ones(36,6)*1000,
            curves, capacities=[10200,16000,13860,6400,22500,2715])
        self.assertTrue(r['checks']['terminal'])
        self.assertTrue(r['checks']['power'])

    def check(self, q, target, **extra):
        return operating_checks(q, torch.zeros(1,2), None, torch.ones(1,1),
            target, torch.full((2,),1e6), torch.zeros_like(q), torch.ones_like(q)*2,
            torch.zeros(2,1), torch.ones(2,1)*2, IdentityCurves(), **extra)

    def test_stationary_flow_can_fail_terminal_requirement(self):
        r = self.check(torch.ones(1,2,1), torch.ones(1,1))
        self.assertFalse(r['checks']['terminal'])
        self.assertFalse(r['feasible'])

    def test_terminal_check_is_level_not_storage_numerical_tolerance(self):
        r = self.check(torch.zeros(1,2,1), torch.tensor([[1.005]]))
        self.assertTrue(r['feasible'])

    def test_path_bounds_have_no_extra_tolerance(self):
        q = torch.zeros(1,2,1)
        r = self.check(q, torch.ones(1,1), level_limits={
            'H_min': torch.zeros(2,1), 'H_max': torch.ones(2,1)-1e-7})
        self.assertFalse(r['checks']['water_level'])

    def test_cascade_uses_same_period_upstream_release(self):
        q = torch.tensor([[[2.,3.],[4.,5.]]])
        incoming, _ = cascade_state(q, torch.tensor([[8.,9.]]),
            torch.ones(1,2,1), torch.zeros(1,2), torch.ones(2))
        self.assertTrue(torch.equal(incoming, torch.tensor([[[8.,3.],[9.,5.]]])))


if __name__ == '__main__': unittest.main()
