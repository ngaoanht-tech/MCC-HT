"""Paper evaluation definitions and bounded cascade updating (no optimization)."""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np


def require_year_files(directory, years):
    missing = [str(Path(directory) / f'{int(y)}.csv') for y in years
               if not (Path(directory) / f'{int(y)}.csv').is_file()]
    if missing:
        raise FileNotFoundError('Requested experiment years are incomplete: ' + ', '.join(missing))


def validate_year_split(train, validation):
    overlap = set(train) & set(validation)
    if overlap:
        raise ValueError(f'Training and validation years overlap: {sorted(overlap)}')


def run_cascade_updates(initial, step, feasible, max_iterations=3):
    if int(max_iterations) not in (2, 3):
        raise ValueError('The paper permits two or three cascade updates.')
    current = initial
    for iteration in range(1, int(max_iterations) + 1):
        current = step(current)
        if iteration >= 2:
            passed = bool(feasible(current))
            if passed:
                return current, iteration, True
    return current, iteration, False


def evaluate_schedules(levels, flows, powers, *, target_levels, level_min,
                       level_max, flow_min, flow_max, power_min, power_max):
    """Inputs [year, period, reservoir]; bounds broadcast to that shape.

    FR is terminal-only annual compliance. Complete final feasibility is
    reported separately. These definitions apply equally to pre/post stages.
    """
    h, q, p = [np.asarray(x, dtype=float) for x in (levels, flows, powers)]
    if h.ndim != 3 or q.shape != h.shape or p.shape != h.shape:
        raise ValueError('Expected matching [year, period, reservoir] arrays.')
    if not h.size or not all(np.isfinite(x).all() for x in (h, q, p)):
        raise ValueError('Schedules must be nonempty and finite.')
    target = np.asarray(target_levels, dtype=float)
    if not np.isfinite(target).all():
        raise ValueError('Terminal targets must be finite.')
    checks = []
    result = {}
    for label, values, lo, hi in (
        ('water_level', h, level_min, level_max),
        ('release', q, flow_min, flow_max),
        ('power', p, power_min, power_max)):
        lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo > hi):
            raise ValueError('Bounds must be finite and ordered.')
        lower, upper = values < lo, values > hi
        result[label + '_lower_violation_rate'] = float(lower.mean() * 100)
        result[label + '_upper_violation_rate'] = float(upper.mean() * 100)
        result[label + '_violation_rate'] = float((lower | upper).mean() * 100)
        checks.append((~(lower | upper)).all(axis=(1, 2)))
    terminal_deviation = np.abs(h[:, -1, :] - target)
    terminal_pass = terminal_deviation <= .01
    annual_terminal = terminal_pass.all(axis=1)
    result.update(TPR=float(terminal_pass.mean() * 100),
                  FR=float(annual_terminal.mean() * 100),
                  maximum_terminal_deviation=float(terminal_deviation.max()),
                  final_feasibility_rate=float(np.logical_and.reduce(
                      checks + [annual_terminal]).mean() * 100),
                  evaluated_years=h.shape[0], terminal_checks=int(terminal_pass.size))
    return result


def adjustment_metrics(pre, post):
    pre, post = np.asarray(pre, dtype=float), np.asarray(post, dtype=float)
    if pre.shape != post.shape or pre.ndim != 3 or not pre.size:
        raise ValueError('Expected matching nonempty endpoint [year, period, reservoir] arrays.')
    delta = np.abs(post - pre)
    if not np.isfinite(delta).all():
        raise ValueError('Endpoint differences must be finite.')
    sums = delta.sum(axis=(0, 1))
    share = sums / sums.sum() * 100 if sums.sum() else np.zeros_like(sums)
    return dict(mean_absolute=delta.mean(axis=(0, 1)).tolist(),
                percentile_95=np.percentile(delta, 95, axis=(0, 1)).tolist(),
                maximum=delta.max(axis=(0, 1)).tolist(), share_percent=share.tolist())


def benchmark_runs(years, run, repeats=5, synchronize=lambda: None,
                   clock=time.perf_counter):
    years = list(years)
    if not years or repeats != 5:
        raise ValueError('Provide years and exactly five repeats per year.')
    results = []
    for year in years:
        seconds = []
        for _ in range(repeats):
            synchronize()
            start = clock()
            run(year)
            synchronize()
            seconds.append(float(clock() - start))
        results.append(dict(year=int(year), seconds=seconds,
                            mean_seconds=float(np.mean(seconds))))
    return dict(years=results, repeats_per_year=5,
                overall_mean_seconds=float(np.mean([x['mean_seconds'] for x in results])))
