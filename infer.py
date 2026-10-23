#!/usr/bin/env python3
"""Generate configured test years using a previously trained checkpoint."""
from config_loader import get_config
from multiscale_predictor import run_multiscale_inference
if __name__ == '__main__':
    cfg = get_config()
    run_multiscale_inference(years=cfg.get('generation', {}).get('years'), samples=1)
