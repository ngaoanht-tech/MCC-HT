"""Small synthetic workflow tests, not a reproduction of manuscript results."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from models import HierarchicalTransformerPredictor
from multiscale_predictor import _run_self_consistent_refinement
from constraint.differentiable_joint_projection import project_joint_schedule
from paper_evaluate import evaluate_directory


class Curves:
    def v2h_all(self, x): return x + 100
    def q2zt_all(self, x): return torch.zeros_like(x)


class PipelineTests(unittest.TestCase):
    def test_full_model_icr_outer_and_retained_joint_backward(self):
        torch.manual_seed(123)
        model = HierarchicalTransformerPredictor(input_dim=22, output_dim=6,
            sequence_length=36, output_sequence_length=36, dropout=0.,
            enable_multiscale=False, enable_sequence_decoding=True)
        model.eval()
        x = torch.randn(1,36,22)
        qmin, qmax = torch.zeros(1,36,6), torch.ones(1,36,6)*20
        incoming = torch.ones(1,36,6)*10
        initial = torch.ones(1,6)*5
        vmin, vmax = torch.zeros(36,6), torch.ones(36,6)*10
        seconds = torch.ones(36)*1e6
        q, logits, raw = model(x, q_min=qmin, q_max=qmax, q_in=incoming,
            V0=initial, V_target=initial, delta_t=seconds, V_min=vmin,
            V_max=vmax, return_logits=True)
        self.assertEqual(raw.shape, (1,36,6))
        out = _run_self_consistent_refinement(q_phys=q, raw_logits=logits,
            model=model, head_inflow=incoming[...,0], interval_inflow=torch.zeros(1,36,5),
            q_min=qmin, q_max=qmax, V0=initial, V_target=initial,
            delta_t=seconds, V_min=vmin, V_max=vmax, sc_iters=3, curves=Curves())
        self.assertIn(out['iterations'], (2,3))
        final, v = project_joint_schedule(out['q_phys'], incoming[...,0],
            torch.zeros(1,36,5), initial, initial, qmin, qmax, vmin, vmax, seconds)
        self.assertTrue(torch.allclose(v[:,-1], initial, atol=1e-4))
        final[0,0,0].backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_stage_archive_aggregation_and_missing_year_error(self):
        with tempfile.TemporaryDirectory() as d:
            arrays = dict(target_levels=np.ones(2), level_min=np.zeros((2,2)),
                level_max=np.ones((2,2))*2, flow_min=np.zeros((2,2)),
                flow_max=np.ones((2,2))*2, power_max=np.ones(2)*10, seconds=np.ones(2)*3600)
            for stage in ('pre','post_icr','final'):
                arrays[stage+'_levels'] = np.ones((2,2))
                arrays[stage+'_flow'] = np.ones((2,2))
                arrays[stage+'_power'] = np.ones((2,2))*3
            np.savez(Path(d)/'paper_metrics_inputs_2022.npz', **arrays)
            m = evaluate_directory(d, [2022])
            self.assertEqual(m['stages']['pre']['FR'], 100.)
            self.assertEqual(m['stages']['final']['annual_energy_MWh'], [12.])
            json.dumps(m, allow_nan=False)
            with self.assertRaises(FileNotFoundError): evaluate_directory(d, [2021,2022])


if __name__ == '__main__': unittest.main()
