"""Measure five complete annual inference runs per year; no model training."""
import argparse
import json
from pathlib import Path
import torch
from paper_contract import benchmark_runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--years', nargs='+', type=int, default=list(range(2015, 2023)))
    p.add_argument('--output-dir', default='benchmark_results')
    p.add_argument('--report', default='runtime_measurements.json')
    args = p.parse_args()
    from multiscale_predictor import run_multiscale_inference
    sync = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
    def run(year):
        run_multiscale_inference(years=[year], samples=1, output_dir=args.output_dir)
    report = benchmark_runs(args.years, run, synchronize=sync)
    report.update(scope='Full inference entry including loading, ICR, cascade updates, retained joint projection and stage exports; training excluded.',
                  device='cuda' if torch.cuda.is_available() else 'cpu',
                  joint_projection_retained=True,
                  note='Fresh measurements for this code version; not verification of the published runtime.')
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
