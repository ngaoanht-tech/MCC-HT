#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-process generated schedules to add level/power/energy columns.

Inputs
- diverse_results/diverse_schedules_YYYY.csv
- shuxing/curves/{水库}.csv (level-storage curve)
- shuxing/{水库}出库流量-下游水位曲线.csv (outflow-tailwater curve)
- constraint/constraints_config.json (period durations and reservoir order)

Outputs
- diverse_results/{YYYY}年.xlsx
  - sheet `periods`: original rows + per-reservoir level/power/energy columns
  - sheet `summary`: annual energy summary by scheme
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

SCRIPT_DIR = Path(__file__).resolve().parent.parent
CURVES_DIR = SCRIPT_DIR / "shuxing" / "curves"
SHUXING_DIR = SCRIPT_DIR / "shuxing"
RESULTS_DIR = SCRIPT_DIR / "diverse_results"
CONSTRAINT_JSON = SCRIPT_DIR / "constraint" / "constraints_config.json"
PLANT_FEATURES_CANDIDATES = [
    SHUXING_DIR / "电站特性.csv",
    SHUXING_DIR / "金沙江四库特征参数.csv",
]
DEFAULT_RESERVOIRS = ["乌东德", "白鹤滩", "溪洛渡", "向家坝", "三峡", "葛洲坝"]


def _read_csv(path: Path) -> pd.DataFrame:
    last_exc: Optional[Exception] = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"failed to read csv {path}: {last_exc}")


def _load_reservoir_order() -> List[str]:
    try:
        cfg = json.loads(CONSTRAINT_JSON.read_text(encoding="utf-8-sig"))
        names = cfg.get("system_config", {}).get("reservoirs")
        if isinstance(names, list) and names:
            return [str(n) for n in names]
    except Exception:
        pass
    return list(DEFAULT_RESERVOIRS)


def _time_steps_seconds() -> np.ndarray:
    try:
        cfg = json.loads(CONSTRAINT_JSON.read_text(encoding="utf-8-sig"))
        wb = cfg.get("water_balance_config", {})
        days = wb.get("time_step_days", [])
        spd = float(wb.get("seconds_per_day", 86400.0))
        if isinstance(days, list) and days:
            return (np.asarray(days, dtype=np.float64) * spd).astype(np.float64)
    except Exception:
        pass
    return np.ones((36,), dtype=np.float64) * 10.0 * 86400.0


def _select_name_col(columns: Sequence[str]) -> Optional[str]:
    for cand in ("水库名称", "电站名称", "水库", "电站"):
        if cand in columns:
            return cand
    return None


def _select_capacity_col(columns: Sequence[str]) -> Optional[str]:
    for cand in ("装机容量(MW)", "装机容量", "总装机容量", "总装机"):
        if cand in columns:
            return cand
    return next((c for c in columns if "装机" in c), None)


def _load_capacity_map() -> Dict[str, float]:
    capacity: Dict[str, float] = {}
    path = next((p for p in PLANT_FEATURES_CANDIDATES if p.exists()), None)
    if path is None:
        return capacity

    try:
        df = _read_csv(path)
        cols = [str(c).strip() for c in df.columns]
        name_col = _select_name_col(cols)
        cap_col = _select_capacity_col(cols)
        if not name_col or not cap_col:
            return capacity
        df[name_col] = df[name_col].astype(str).str.strip()
        for _, row in df.iterrows():
            name = str(row[name_col]).strip()
            if not name:
                continue
            try:
                cap_val = float(row[cap_col])
            except Exception:
                continue
            if cap_val > 0:
                capacity[name] = cap_val
    except Exception:
        return {}
    return capacity


def _select_curve_columns(df: pd.DataFrame, kind: str) -> Tuple[pd.Series, pd.Series]:
    cols = [str(c).strip() for c in df.columns]
    if not cols:
        raise ValueError("empty curve file")
    if kind == "HV":
        level_col = next((c for c in cols if ("水位" in c or c.upper().startswith("H") or c.upper().startswith("Z"))), cols[0])
        vol_col = next((c for c in cols if ("库容" in c or c.upper().startswith("V"))), cols[1] if len(cols) > 1 else cols[0])
        return df[level_col], df[vol_col]
    if kind == "QTW":
        flow_col = next((c for c in cols if ("出库" in c or c.upper().startswith("Q"))), cols[0])
        tail_col = next((c for c in cols if ("下游水位" in c or "水位" in c or c.upper().startswith("H") or c.upper().startswith("Z"))), cols[-1])
        return df[flow_col], df[tail_col]
    raise ValueError(f"unknown curve kind: {kind}")


def _interp_1d(x: np.ndarray, y: np.ndarray, *, extrapolate: bool = False):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        raise ValueError("curve needs at least two finite points")
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    x_unique, idx = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[idx]
    if x_unique.size < 2:
        raise ValueError("curve needs at least two unique x points")
    if extrapolate:
        return interp1d(x_unique, y_unique, kind="linear", fill_value="extrapolate", assume_sorted=True)
    return interp1d(
        x_unique,
        y_unique,
        kind="linear",
        bounds_error=False,
        fill_value=(float(y_unique[0]), float(y_unique[-1])),
        assume_sorted=True,
    )


def _build_hv_and_qtw_interps(reservoirs: List[str]) -> Tuple[Dict[str, interp1d], Dict[str, interp1d]]:
    v2h: Dict[str, interp1d] = {}
    q2tail: Dict[str, interp1d] = {}
    for name in reservoirs:
        hv_candidates = [CURVES_DIR / f"{name}.csv", CURVES_DIR / f"{name}水位-库容曲线.csv"]
        hv_path = next((p for p in hv_candidates if p.exists()), None)
        if hv_path is None:
            raise FileNotFoundError(f"missing level-storage curve for {name}: tried {hv_candidates}")
        df_hv = _read_csv(hv_path)
        level_s, vol_s = _select_curve_columns(df_hv, "HV")
        h = pd.to_numeric(level_s, errors="coerce").to_numpy(np.float64)
        v = pd.to_numeric(vol_s, errors="coerce").to_numpy(np.float64)
        v2h[name] = _interp_1d(v, h, extrapolate=True)

        qtw_candidates = [
            SHUXING_DIR / f"{name}出库流量-下游水位曲线.csv",
            SHUXING_DIR / f"{name}流量-下游水位曲线.csv",
        ]
        qtw_path = next((p for p in qtw_candidates if p.exists()), None)
        if qtw_path is None:
            raise FileNotFoundError(f"missing outflow-tailwater curve for {name}: tried {qtw_candidates}")
        df_qtw = _read_csv(qtw_path)
        q_s, z_s = _select_curve_columns(df_qtw, "QTW")
        q = pd.to_numeric(q_s, errors="coerce").to_numpy(np.float64)
        z = pd.to_numeric(z_s, errors="coerce").to_numpy(np.float64)
        q2tail[name] = _interp_1d(q, z, extrapolate=False)
    return v2h, q2tail


def _find_columns(df: pd.DataFrame, reservoir: str) -> Tuple[str, str]:
    cols = [str(c) for c in df.columns]
    flow_col = next((c for c in cols if reservoir in c and "出库" in c and "入库" not in c), None)
    if flow_col is None:
        raise KeyError(f"missing outflow column for {reservoir}")
    storage_col = next((c for c in cols if reservoir in c and "库容" in c), None)
    if storage_col is None:
        raise KeyError(f"missing storage column for {reservoir}")
    return flow_col, storage_col


def process_year(year: int, eta: float = 0.88) -> Path:
    in_csv = RESULTS_DIR / f"diverse_schedules_{year}.csv"
    if not in_csv.exists():
        raise FileNotFoundError(str(in_csv))
    df = _read_csv(in_csv)

    reservoirs = _load_reservoir_order()
    capacity_map = _load_capacity_map()
    v2h, q2tw = _build_hv_and_qtw_interps(reservoirs)
    dt_vec = _time_steps_seconds()
    dt = np.array(dt_vec, dtype=np.float64)

    rho = 1000.0
    g = 9.81
    eta_eff = float(eta if eta <= 1.0 else eta / 100.0)

    total_power_col = "TOTAL_出力(MW)"
    total_energy_col = "TOTAL_能量(MWh)"
    seconds_col = "_period_seconds"

    for name in reservoirs:
        df[f"{name}_水位(m)"] = np.nan
        df[f"{name}_出力(MW)"] = np.nan
        df[f"{name}_能量(MWh)"] = np.nan
        df[f"{name}_发电流量(m3/s)"] = np.nan
        df[f"{name}_弃水流量(m3/s)"] = np.nan
    df[total_power_col] = 0.0
    df[total_energy_col] = 0.0

    if "period" in df.columns:
        try:
            per_idx = pd.to_numeric(df["period"], errors="coerce").fillna(0).astype(int) - 1
            per_idx = per_idx.clip(lower=0, upper=max(0, len(dt) - 1)).to_numpy()
            seconds_arr = dt[per_idx] if len(dt) else np.full(len(df), 10.0 * 86400.0, dtype=np.float64)
        except Exception:
            seconds_arr = np.full(len(df), dt[0] if len(dt) else 10.0 * 86400.0, dtype=np.float64)
    else:
        seconds_arr = np.full(len(df), dt[0] if len(dt) else 10.0 * 86400.0, dtype=np.float64)
    df[seconds_col] = seconds_arr

    energy_cols: List[str] = []
    power_cols: List[str] = []
    for name in reservoirs:
        try:
            flow_col, storage_col = _find_columns(df, name)
        except KeyError:
            continue

        v = pd.to_numeric(df[storage_col], errors="coerce").to_numpy(np.float64)
        q = pd.to_numeric(df[flow_col], errors="coerce").to_numpy(np.float64)
        h = v2h[name](v)
        zt = q2tw[name](q)
        hnet = np.maximum(h - zt, 0.0)
        p_mw = rho * g * eta_eff * q * hnet / 1.0e6

        cap = capacity_map.get(name)
        if cap is not None and np.isfinite(cap) and cap > 0:
            p_mw = np.clip(p_mw, 0.0, cap)

        denom = rho * g * eta_eff * np.maximum(hnet, 1e-6)
        q_gen = np.where(denom > 0.0, (p_mw * 1.0e6) / denom, 0.0)
        q_gen = np.minimum(q_gen, np.maximum(q, 0.0))
        q_spill = np.maximum(q - q_gen, 0.0)
        e_mwh = p_mw * (seconds_arr / 3600.0)

        p_col = f"{name}_出力(MW)"
        e_col = f"{name}_能量(MWh)"
        df[f"{name}_水位(m)"] = h
        df[p_col] = p_mw
        df[e_col] = e_mwh
        df[f"{name}_发电流量(m3/s)"] = q_gen
        df[f"{name}_弃水流量(m3/s)"] = q_spill
        power_cols.append(p_col)
        energy_cols.append(e_col)

    if power_cols:
        df[total_power_col] = df[power_cols].sum(axis=1)
    if energy_cols:
        df[total_energy_col] = df[energy_cols].sum(axis=1)
    else:
        df[total_energy_col] = df[total_power_col] * (seconds_arr / 3600.0)

    if "scheme" in df.columns:
        agg_cols = [total_energy_col] + energy_cols
        summary = df.groupby("scheme", as_index=False)[agg_cols].sum()
    else:
        summary = pd.DataFrame({"scheme": ["scheme_01"], total_energy_col: [float(df[total_energy_col].sum())]})

    summary = summary.rename(columns={total_energy_col: "年总能量(MWh)"})
    for col in energy_cols:
        summary = summary.rename(columns={col: col.replace("能量(MWh)", "年总能量(MWh)")})
    summary["年总能量(GWh)"] = summary["年总能量(MWh)"].astype(float) / 1000.0
    summary["年总能量(亿kWh)"] = summary["年总能量(MWh)"].astype(float) / 1.0e5

    out_xlsx = RESULTS_DIR / f"{year}年.xlsx"
    df_to_save = df.drop(columns=[seconds_col], errors="ignore")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_to_save.to_excel(writer, sheet_name="periods", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
    return out_xlsx


def _discover_years() -> List[int]:
    years: List[int] = []
    for path in RESULTS_DIR.glob("diverse_schedules_*.csv"):
        try:
            years.append(int(path.stem.split("_")[-1]))
        except Exception:
            continue
    return sorted(set(years))


def main() -> None:
    parser = argparse.ArgumentParser(description="Add level/power fields and summarize annual energy.")
    parser.add_argument("--years", nargs="*", type=int, help="Years to process (default: infer from files)")
    parser.add_argument("--eta", type=float, default=0.88, help="Generator efficiency (0.88 or 88)")
    args = parser.parse_args()

    years = list(dict.fromkeys(int(y) for y in args.years)) if args.years else _discover_years()
    if not years:
        raise SystemExit("No diverse_schedules_YYYY.csv files found.")

    for year in years:
        out_path = process_year(int(year), eta=float(args.eta))
        print(f"[postprocess] {year} -> {out_path}")


if __name__ == "__main__":
    main()

