# MCC-HT

MCC-HT is a Transformer-based scheduling framework for a six-reservoir cascade:
乌东德, 白鹤滩, 溪洛渡, 向家坝, 三峡, and 葛洲坝.  The repository contains the
paper baseline code, constraint data, test inputs, and small example outputs.

## What Is Included

- `multiscale_predictor.py`: model training and the main annual inference entry point.
- `models.py`, `losses.py`, `feature_engineering.py`, `data.py`: model, objectives,
  features, and annual dataset preparation.
- `constraint/`: release, storage, and water-level constraint utilities.
- `shuxing/`: reservoir curves, plant characteristics, and constraint tables.
- `train/`, `test/`, `real_results/`: input data used by the baseline workflow.
- `utils/`: post-processing, feasibility checks, and evaluation utilities.
- `examples/baseline_results/`: one generated schedule and post-processed workbook
  for each year from 2015 to 2022.

Model checkpoints, preprocessing transforms, training logs, bulk generated results,
and experiment archives are intentionally excluded from Git.  They must be supplied
locally under `results/` before inference or training.

## Environment

Python 3.10 or later is recommended.  Install the dependencies in an isolated
environment:

```bash
pip install -r requirements.txt
```

For GPU inference, install the PyTorch build matching the local CUDA runtime from
the official PyTorch installation instructions.

## Required Local Artifacts

Before inference, place the baseline artifacts in `results/`:

```text
results/
  multiscale_best_model.pth
  multiscale_hyperparameters.json
  annual_transforms.pkl
```

These files are excluded because they are binary experiment artifacts.  Their names
must match `config.yaml` and the inference code.

## Run Inference

One-click inference for the reference year (2022):

```bash
python infer.py
```

Or manually:

```bash
python -X utf8 -c "from multiscale_predictor import run_multiscale_inference; run_multiscale_inference(years=[2022], samples=1)"
```

## Post-Process Results

```bash
python -X utf8 utils/postprocess_power_level.py --years 2022
python -X utf8 -c "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'utils'); import feasibility_report as fr; fr.YEARS=[2022]; fr.main()"
```

## Data Availability

The repository includes **6 reference years** (1959, 1961–1964 in `train/`; 2022 in `test/`)
for format validation and code testing. The full 1959–2022 hydrological dataset used in
the paper is subject to institutional data-sharing policies and is available from the
corresponding author upon reasonable request.

A placeholder data generator is provided for pipeline validation:
```bash
python generate_dummy.py
```

## Output Units

- Release and inflow: `m3/s`.
- Storage: `1e8 m3` (亿立方米).
- Water level: `m`.
- Power: `MW`.
- Energy: `MWh`, `GWh`, and `1e8 kWh` in the post-processing summary.

## Reproducibility Notes

This repository is the paper baseline release.  It preserves the original workflow
and example outputs; it is not a claim that later experimental branches or audit
tools are included here.  Treat the CSV schedule as the direct inference output and
the Excel workbook as a derived post-processing artifact.

## Repository Layout

```text
MCC-HT/
  constraint/                 Constraint and water-level conversion code
  shuxing/                    Curves and plant/constraint data
  test/                       Annual inference input (2022 reference)
  train/                      Historical training inputs (5 reference years)
  utils/                      Post-processing and evaluation scripts
  README.md                   This guide
  config.yaml                 Configuration (synced with paper Table 1)
  train.py                    One-click training
  infer.py                    One-click inference
  generate_dummy.py           Placeholder data generator
```

## Key Parameters

All parameters in `config.yaml` match the paper:

| Category | Parameter | Value |
|:---------|:----------|:------|
| Model | d_model / nhead / layers | 144 / 12 / 4 |
| Model | FFN / Dropout | 576 / 0.2 |
| Training | Optimizer | AdamW (lr=1e-3, wd=1e-4) |
| Training | Max epochs / patience | 500 / 120 |
| Training | Gradient clipping | 0.8 |
| ICR | Self-consistent iterations | 3 |
| Loss | λ_flow | 0.35 → 0.05 (epoch 0–45) |
| Loss | λ_term | 1.2 → 2.4 (epoch 0–32) |
| Loss | λ_pow | 0.0 → 0.04 (epoch 18–80) |
| Loss | λ_smooth | 0.20 (fixed) |
