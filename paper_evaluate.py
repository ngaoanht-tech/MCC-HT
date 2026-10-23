"""Aggregate stage-labelled metrics from inference exports, without rerunning a model."""
import argparse
import json
from pathlib import Path
import numpy as np
from paper_contract import evaluate_schedules, adjustment_metrics


def evaluate_directory(directory, years):
    directory = Path(directory)
    years = list(years)
    if not years or len(set(years)) != len(years):
        raise ValueError('Provide distinct evaluation years.')
    records = []
    for year in years:
        path = directory / f'paper_metrics_inputs_{year}.npz'
        if not path.exists():
            raise FileNotFoundError(f'Missing {path}; partial years are not silently omitted.')
        with np.load(path, allow_pickle=False) as z:
            records.append({k: z[k] for k in z.files})
    stack = lambda key: np.stack([x[key] for x in records])
    result = dict(years=years, scheme='scheme_01', joint_projection_retained=True,
                  stages={}, adjustments={})
    for stage in ('pre', 'post_icr', 'final'):
        m = evaluate_schedules(stack(stage+'_levels'), stack(stage+'_flow'),
            stack(stage+'_power'), target_levels=stack('target_levels'),
            level_min=stack('level_min'), level_max=stack('level_max'),
            flow_min=stack('flow_min'), flow_max=stack('flow_max'),
            power_min=0., power_max=stack('power_max')[:, None, :])
        energy = (stack(stage+'_power') * stack('seconds')[..., None] / 3600.).sum(axis=(1,2))
        m['annual_energy_MWh'] = energy.tolist()
        m['mean_annual_energy_1e8_kWh'] = float(energy.mean() / 1e5)
        result['stages'][stage] = m
    result['adjustments']['pre_to_post_icr'] = adjustment_metrics(stack('pre_flow'), stack('post_icr_flow'))
    result['adjustments']['pre_to_final_with_joint'] = adjustment_metrics(stack('pre_flow'), stack('final_flow'))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', default='diverse_results')
    p.add_argument('--years', nargs='+', type=int, default=list(range(2015, 2023)))
    p.add_argument('--output', default='paper_metrics_summary.json')
    args = p.parse_args()
    result = evaluate_directory(args.directory, args.years)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(path.resolve())


if __name__ == '__main__': main()
