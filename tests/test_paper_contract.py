import unittest
import numpy as np
from paper_contract import run_cascade_updates, evaluate_schedules, adjustment_metrics, benchmark_runs, validate_year_split, require_year_files


class PaperContractTests(unittest.TestCase):
    def test_overlapping_training_validation_is_rejected(self):
        with self.assertRaises(ValueError): validate_year_split([1959,1963], [1963])

    def test_missing_experiment_year_is_not_silently_skipped(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError): require_year_files(d, [2022])

    def test_stationary_but_infeasible_requires_third_round(self):
        q, rounds, passed = run_cascade_updates(1, lambda q: q, lambda q: False)
        self.assertEqual((q, rounds, passed), (1, 3, False))

    def test_feasible_after_second_round_stops_even_if_changing(self):
        q, rounds, passed = run_cascade_updates(0, lambda q: q + 10, lambda q: q >= 20)
        self.assertEqual((q, rounds, passed), (20, 2, True))

    def test_terminal_rate_is_distinct_from_path_feasibility(self):
        levels = np.array([[[2., 1.], [1., 1.]], [[1., 1.], [1., 1.02]]])
        flow = np.ones_like(levels)
        m = evaluate_schedules(levels, flow, np.zeros_like(levels),
            target_levels=np.ones(2), level_min=0, level_max=1.5,
            flow_min=0, flow_max=2, power_min=0, power_max=1)
        self.assertEqual(m['FR'], 50.)
        self.assertEqual(m['TPR'], 75.)
        self.assertEqual(m['final_feasibility_rate'], 0.)
        self.assertEqual(m['water_level_upper_violation_rate'], 12.5)

    def test_adjustments_compare_endpoints_not_iteration_sums(self):
        pre = np.zeros((1, 2, 2)); post = np.array([[[1., 3.], [1., 3.]]])
        m = adjustment_metrics(pre, post)
        self.assertEqual(m['mean_absolute'], [1., 3.])
        self.assertEqual(m['share_percent'], [25., 75.])

    def test_five_runs_then_equal_year_mean(self):
        clock = [0.]
        def run(year): clock[0] += year
        m = benchmark_runs([1, 3], run, repeats=5, clock=lambda: clock[0])
        self.assertEqual(m['overall_mean_seconds'], 2.)
        self.assertEqual(len(m['years'][0]['seconds']), 5)

    def test_nonfinite_schedule_is_not_accepted(self):
        a = np.array([[[float('nan')]]])
        with self.assertRaises(ValueError):
            evaluate_schedules(a, a, a, target_levels=[0], level_min=0,
                level_max=1, flow_min=0, flow_max=1, power_min=0, power_max=1)


if __name__ == '__main__': unittest.main()
