#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute year-level metrics summaries without plotting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / 'diverse_results'
REAL_RESULTS_DIR = BASE_DIR / 'real_results'
FIGS_DIR = RESULTS_DIR / 'figs'
RESERVOIRS = ['乌东德', '白鹤滩', '溪洛渡', '向家坝', '三峡', '葛洲坝']


def _norm(text: object) -> str:
    s = str(text or '').strip().lower()
    repl = {
        '（': '(',
        '）': ')',
        ' ': '',
        'm³': 'm3',
        'm^3': 'm3',
        'm鲁': 'm3',
    }
    for old, new in repl.items():
        s = s.replace(old, new)
    return s


def _find_column(df: pd.DataFrame, keywords: Sequence[object], default: str | None = None) -> str:
    groups: List[List[str]] = []
    for token in keywords:
        if isinstance(token, (list, tuple, set)):
            groups.append([_norm(x) for x in token])
        else:
            groups.append([_norm(token)])
    for col in map(str, df.columns):
        col_norm = _norm(col)
        if all(any(g and g in col_norm for g in group) for group in groups):
            return col
    if default is not None:
        return default
    raise KeyError(str(keywords))


def _load_schedule(year: int) -> pd.DataFrame:
    xlsx = RESULTS_DIR / f'{year}年.xlsx'
    if xlsx.exists():
        df = pd.read_excel(xlsx, sheet_name='periods')
    else:
        df = pd.read_csv(RESULTS_DIR / f'diverse_schedules_{year}.csv', encoding='utf-8-sig')
    if 'scheme' in df.columns:
        df = df[df['scheme'] == 'scheme_01'].copy()
    if 'period' in df.columns:
        df['period'] = pd.to_numeric(df['period'], errors='coerce')
        df = df.sort_values('period')
    return df.reset_index(drop=True)


def _load_real(year: int) -> pd.DataFrame:
    df = pd.read_csv(REAL_RESULTS_DIR / f'{year}.csv', encoding='utf-8-sig')
    if 'period' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'period'})
    df['period'] = pd.to_numeric(df['period'], errors='coerce')
    return df.sort_values('period').reset_index(drop=True)


def _stat_summary(model: pd.Series, real: pd.Series) -> Dict[str, float]:
    mask = model.notna() & real.notna()
    if int(mask.sum()) < 2:
        return {'rmse': float('nan'), 'nse': float('nan')}
    m = model[mask].to_numpy(dtype=float)
    r = real[mask].to_numpy(dtype=float)
    diff = m - r
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    denom = np.sum((r - np.mean(r)) ** 2)
    nse = float('nan') if denom <= 1e-8 else float(1.0 - np.sum(diff ** 2) / (denom + 1e-8))
    return {'rmse': rmse, 'nse': nse}


def _energy_to_mwh(series: pd.Series, column_name: str) -> float:
    values = pd.to_numeric(series, errors='coerce').to_numpy(dtype=float)
    total = float(np.nansum(values))
    name = str(column_name)
    if '万kWh' in name:
        return total * 10.0
    if '亿kWh' in name:
        return total * 100000.0
    if 'GWh' in name:
        return total * 1000.0
    return total


def compute_metrics(year: int) -> Dict[str, float]:
    model = _load_schedule(year)
    real = _load_real(year)
    dt_hours = np.full(len(model), 24.0 * 10.0, dtype=float)

    flow_diffs = []
    level_diffs = []
    terminal_diffs: List[float] = []
    flow_nse_values: List[float] = []
    level_nse_values: List[float] = []
    energy_model = 0.0
    energy_real = 0.0
    has_power = False

    for res in RESERVOIRS:
        try:
            model_flow_col = _find_column(model, [res, '出库', '流量'])
            real_flow_col = _find_column(real, [res, '出库', '流量'])
        except KeyError:
            continue
        flow_model = pd.to_numeric(model[model_flow_col], errors='coerce')
        flow_real = pd.to_numeric(real[real_flow_col], errors='coerce')
        min_len = min(len(flow_model), len(flow_real))
        flow_model = flow_model.iloc[:min_len]
        flow_real = flow_real.iloc[:min_len]
        flow_diffs.append((flow_model - flow_real).to_numpy(dtype=float))
        flow_stats = _stat_summary(flow_model, flow_real)
        if not np.isnan(flow_stats['nse']):
            flow_nse_values.append(flow_stats['nse'])

        try:
            model_level_col = _find_column(model, [res, '水位'])
            real_level_col = _find_column(real, [res, '水位'])
            level_model = pd.to_numeric(model[model_level_col], errors='coerce').iloc[:min_len]
            level_real = pd.to_numeric(real[real_level_col], errors='coerce').iloc[:min_len]
            level_diffs.append((level_model - level_real).to_numpy(dtype=float))
            if not np.isnan(level_model.iloc[-1]) and not np.isnan(level_real.iloc[-1]):
                terminal_diffs.append(float(abs(level_model.iloc[-1] - level_real.iloc[-1])))
            level_stats = _stat_summary(level_model, level_real)
            if not np.isnan(level_stats['nse']):
                level_nse_values.append(level_stats['nse'])
        except Exception:
            pass

        try:
            model_power_col = _find_column(model, [res, '出力'])
            power_series = pd.to_numeric(model[model_power_col], errors='coerce').to_numpy(dtype=float)
            energy_model += float(np.nansum(power_series[: len(dt_hours)] * dt_hours[: len(power_series)]))
            has_power = True
        except Exception:
            pass
        try:
            real_energy_col = _find_column(real, [res, ['能量', '发电量']], default='')
            if real_energy_col:
                energy_real += _energy_to_mwh(real[real_energy_col], real_energy_col)
        except Exception:
            pass

    flow_diff = np.concatenate(flow_diffs) if flow_diffs else np.asarray([], dtype=float)
    level_diff = np.concatenate(level_diffs) if level_diffs else np.asarray([], dtype=float)
    flow_rmse = float(np.sqrt(np.nanmean(flow_diff ** 2))) if flow_diff.size else float('nan')
    level_rmse = float(np.sqrt(np.nanmean(level_diff ** 2))) if level_diff.size else float('nan')

    flow_cols = []
    for res in RESERVOIRS:
        try:
            flow_cols.append(_find_column(real, [res, '出库', '流量']))
        except Exception:
            continue
    real_flow_vals = real[flow_cols].to_numpy(dtype=float) if flow_cols else np.asarray([], dtype=float)
    flow_range = float(np.nanmax(real_flow_vals) - np.nanmin(real_flow_vals) + 1e-6) if real_flow_vals.size else float('nan')
    level_cols = [c for c in map(str, real.columns) if '水位' in c]
    real_level_vals = real[level_cols].to_numpy(dtype=float) if level_cols else np.asarray([], dtype=float)
    level_range = float(np.nanmax(real_level_vals) - np.nanmin(real_level_vals) + 1e-6) if real_level_vals.size else float('nan')

    flow_skill = max(0.0, 1.0 - flow_rmse / flow_range) if np.isfinite(flow_rmse) and np.isfinite(flow_range) else 0.0
    level_skill = max(0.0, 1.0 - level_rmse / level_range) if np.isfinite(level_rmse) and np.isfinite(level_range) else 0.0
    energy_skill = max(0.0, 1.0 - abs(energy_model - energy_real) / (energy_real + 1e-6)) if has_power and energy_real > 1e-6 else 0.0
    terminal_skill = max(0.0, 1.0 - float(np.mean(terminal_diffs)) / (level_range + 1e-6)) if terminal_diffs and np.isfinite(level_range) else 0.0

    return {
        'flow_skill': flow_skill,
        'level_skill': level_skill,
        'energy_skill': energy_skill,
        'terminal_skill': terminal_skill,
        'flow_rmse': flow_rmse,
        'level_rmse': level_rmse,
        'energy_model_MWh': energy_model if has_power else float('nan'),
        'energy_real_MWh': energy_real if has_power else float('nan'),
        'terminal_diff_avg': float(np.mean(terminal_diffs)) if terminal_diffs else float('nan'),
        'flow_nse': float(np.nanmean(flow_nse_values)) if flow_nse_values else float('nan'),
        'level_nse': float(np.nanmean(level_nse_values)) if level_nse_values else float('nan'),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute metrics_summary.json for generated schedules.')
    parser.add_argument('--years', nargs='+', type=int, required=True)
    args = parser.parse_args()
    for year in args.years:
        metrics = compute_metrics(int(year))
        out_dir = FIGS_DIR / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'metrics_summary.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
        pd.DataFrame([metrics]).to_csv(out_dir / 'metrics_summary.csv', index=False, encoding='utf-8-sig')
        print(f'[metrics] {year} -> {out_dir / "metrics_summary.json"}')


if __name__ == '__main__':
    main()

