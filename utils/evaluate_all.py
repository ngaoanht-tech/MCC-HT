#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified evaluation utilities (terminal, capacity, feasibility hooks)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

try:  # pragma: no cover
    from .analysis_utils import (
        BASE_DIR,
        DEFAULT_RESERVOIRS,
        RESULTS_DIR,
        find_column,
        load_config,
        load_constraints,
        load_schedule_result,
    )
except ImportError:  # pragma: no cover
    from analysis_utils import (
        BASE_DIR,
        DEFAULT_RESERVOIRS,
        RESULTS_DIR,
        find_column,
        load_config,
        load_constraints,
        load_schedule_result,
    )

import sys

sys.path.append(str(BASE_DIR))
from losses import TorchCurves  # type: ignore


def _reservoir_names(constraints: Dict) -> List[str]:
    names = constraints.get("system_config", {}).get("reservoirs")
    if isinstance(names, list) and names:
        return [str(n) for n in names]
    return list(DEFAULT_RESERVOIRS)


def _dt_seconds(constraints: Dict, periods: int) -> np.ndarray:
    cfg = constraints.get("water_balance_config", {}) or {}
    days = cfg.get("time_step_days", [])
    seconds_per_day = float(cfg.get("seconds_per_day", 86400))
    if days:
        vec = np.asarray(days, dtype=float) * seconds_per_day
    else:
        vec = np.ones((periods,), dtype=float) * 10.0 * seconds_per_day
    if vec.size < periods:
        vec = np.pad(vec, (0, periods - vec.size), mode="edge")
    return vec[:periods]


def _storage_column(df: pd.DataFrame, name: str) -> str:
    return find_column(df, [name, ["库容", "storage", "volume"]])


def _flow_column(df: pd.DataFrame, name: str) -> str:
    return find_column(df, [name, "出库", "流量"])


def _load_test_inflows(year: int, periods: int) -> Tuple[np.ndarray, np.ndarray]:
    test_csv = BASE_DIR / "test" / f"{int(year)}.csv"
    if not test_csv.exists():
        raise FileNotFoundError(str(test_csv))
    df = pd.read_csv(str(test_csv), encoding="utf-8-sig")
    cols = [str(c).strip() for c in df.columns]
    inflow_cols = [c for c in cols if "入库流量" in c]
    interval_cols = [c for c in cols if "区间来水" in c]
    if not inflow_cols:
        raise KeyError("test CSV 缺少入库流量列")
    head = pd.to_numeric(df[inflow_cols[0]].iloc[:periods], errors="coerce").to_numpy(dtype=float)
    if interval_cols:
        interval = df[interval_cols].iloc[:periods].to_numpy(dtype=float)
    else:
        interval = np.zeros((periods, 0), dtype=float)
    return head, interval


def diagnose_need_capacity(
    year: int,
    window_k: int = 10,
    schedule_df: Optional[pd.DataFrame] = None,
    constraints: Optional[Dict] = None,
    config: Optional[Dict] = None,
) -> Dict:
    """Diagnose terminal need vs available capacity."""
    constraints = constraints or load_constraints()
    config = config or load_config()
    names = _reservoir_names(constraints)
    R = len(names)
    schedule = schedule_df if schedule_df is not None else load_schedule_result(year, scheme=None)
    scheme_df = schedule[schedule["scheme"] == "scheme_01"].copy()
    if scheme_df.empty:
        raise RuntimeError(f"{year} 没有 scheme_01 结果")
    periods = int(constraints.get("system_config", {}).get("periods_per_year", len(scheme_df)))
    scheme_df = scheme_df.head(periods).reset_index(drop=True)
    dt = _dt_seconds(constraints, periods)
    dt_vol = dt / 1e8

    flow_cfg = constraints.get("flow_constraints", {}) or {}
    qmin = np.asarray(flow_cfg.get("Qmin", [[0] * periods] * R), dtype=float)
    qmax = np.asarray(flow_cfg.get("Qmax", [[0] * periods] * R), dtype=float)
    qmin = np.resize(qmin, (R, periods))
    qmax = np.resize(qmax, (R, periods))

    target_list = (config.get("constraints", {}) or {}).get("target_storage", [0] * R)
    targets = np.asarray(list(map(float, target_list[:R])), dtype=float)

    last_row = scheme_df.tail(1).reset_index(drop=True)
    V_T = np.zeros((R,), dtype=float)
    for idx, name in enumerate(names):
        col = _storage_column(last_row, name)
        V_T[idx] = float(pd.to_numeric(last_row[col], errors="coerce").iloc[0])
    need = np.abs(targets - V_T)

    K = max(0, min(int(window_k), periods))
    cap_box = ((qmax[:, -K:] - qmin[:, -K:]) * dt_vol[-K:]).sum(axis=1) if K > 0 else np.zeros_like(need)

    # Effective capacity considering inflow contributions
    try:
        head, interval = _load_test_inflows(year, periods)
        flow_cols = [_flow_column(scheme_df, name) for name in names]
        q_out_mat = scheme_df[flow_cols].to_numpy(dtype=float)
        q_in_mat = np.zeros_like(q_out_mat, dtype=float)
        q_in_mat[:, 0] = head[:periods]
        if interval.shape[1] < max(0, R - 1):
            pad = np.zeros((periods, R - 1 - interval.shape[1]), dtype=float)
            interval_eff = np.hstack([interval, pad]) if interval.size else pad
        else:
            interval_eff = interval[:, : max(0, R - 1)]
        for idx in range(1, R):
            q_in_mat[:, idx] = q_out_mat[:, idx - 1] + interval_eff[:, idx - 1]
        if K > 0:
            qmin_K = qmin[:, -K:]
            inflow_excess = np.clip(q_in_mat[-K:, :].T - qmin_K, a_min=0.0, a_max=None)
            cap_eff = (inflow_excess * dt_vol[-K:]).sum(axis=1)
        else:
            cap_eff = np.zeros((R,), dtype=float)
    except Exception:
        cap_eff = cap_box.copy()

    return {
        "year": int(year),
        "window_k": int(K),
        "per_reservoir": {
            names[i]: {
                "V_T": float(V_T[i]),
                "V_target": float(targets[i]),
                "need": float(need[i]),
                "cap_box": float(cap_box[i]),
                "cap_eff": float(cap_eff[i] if i < cap_eff.shape[0] else 0.0),
                "cap_minus_need": float((cap_eff[i] if i < cap_eff.shape[0] else 0.0) - need[i]),
            }
            for i in range(R)
        },
    }


def _resolve_level_targets(names: List[str], term_cfg: Dict) -> torch.Tensor:
    h_cfg = term_cfg.get("h_target", {}) or {}
    default_entry = h_cfg.get("default", {})
    if isinstance(default_entry, dict):
        default_val = default_entry.get("value", 0.0)
    elif default_entry in (None, {}):
        default_val = 0.0
    else:
        default_val = default_entry
    targets = []
    for nm in names:
        entry = h_cfg.get(nm, default_entry)
        if isinstance(entry, dict):
            value = entry.get("value", default_val)
        elif entry in (None, {}):
            value = default_val
        else:
            value = entry
        targets.append(float(value))
    return torch.tensor(targets, dtype=torch.float32)


def _maybe_build_curves(cfg: Dict, names: List[str]) -> Optional[TorchCurves]:
    constraints_cfg = (cfg.get("constraints", {}) or {})
    curves_dir = constraints_cfg.get("curves_dir", "shuxing/curves")
    tailwater_dir = constraints_cfg.get("tailwater_dir", "shuxing")
    try:
        curves = TorchCurves.from_paths(
            str((BASE_DIR / curves_dir).resolve()),
            str((BASE_DIR / tailwater_dir).resolve()),
            names,
        )
        return curves.to(torch.device("cpu"))
    except Exception:
        return None


def _pick_storage_columns(df: pd.DataFrame, preferred_names: Optional[Sequence[str]] = None) -> List[Tuple[str, str]]:
    cols: List[Tuple[str, str]] = []
    if preferred_names:
        for name in preferred_names:
            try:
                cols.append((str(name), _storage_column(df, str(name))))
            except KeyError:
                continue
    if cols:
        return cols
    for name in DEFAULT_RESERVOIRS:
        try:
            cols.append((name, _storage_column(df, name)))
        except KeyError:
            continue
    return cols


def check_terminal_attainment(
    year: int,
    tol: float = 1e-3,
    schedule_df: Optional[pd.DataFrame] = None,
    config: Optional[Dict] = None,
) -> Dict:
    """Check terminal storage/level attainment for all schemes."""
    df = schedule_df if schedule_df is not None else load_schedule_result(year, scheme=None)
    if "scheme" not in df.columns:
        raise KeyError("生成结果缺少 scheme 列")
    cfg = config or load_config()
    constraints = load_constraints()
    names = _reservoir_names(constraints)
    res_pairs = _pick_storage_columns(df, names)
    if not res_pairs:
        raise KeyError("未找到库容列，无法判断末水位达标")

    target_list = (cfg.get("constraints", {}) or {}).get("target_storage", [])
    V_target = np.asarray(list(map(float, target_list[: len(res_pairs)])), dtype=float)
    if V_target.size < len(res_pairs):
        pad = np.zeros((len(res_pairs) - V_target.size,), dtype=float)
        V_target = np.concatenate([V_target, pad])

    term_cfg = (cfg.get("loss", {}) or {}).get("terminal", {}) or {}
    term_mode = str(term_cfg.get("mode", "volume_window"))
    h_tol = float(term_cfg.get("h_tolerance", 0.3))

    curve = _maybe_build_curves(cfg, [name for name, _ in res_pairs])
    V_target_tensor = torch.tensor(V_target, dtype=torch.float32)
    if term_mode == "level_window" and curve is not None:
        H_target = _resolve_level_targets([name for name, _ in res_pairs], term_cfg)
        H_lo_ref = H_target - h_tol
        H_hi_ref = H_target + h_tol
        V_lo_ref = curve.h2v_all(H_lo_ref)
        V_hi_ref = curve.h2v_all(H_hi_ref)
    else:
        V_lo_ref = V_target_tensor - tol
        V_hi_ref = V_target_tensor + tol
        if curve is not None:
            H_lo_ref = curve.v2h_all(V_lo_ref)
            H_hi_ref = curve.v2h_all(V_hi_ref)
        else:
            H_lo_ref = torch.zeros_like(V_lo_ref)
            H_hi_ref = torch.zeros_like(V_hi_ref)

    results: Dict[str, Dict] = {}
    pass_count = 0
    for scheme_name, grp in df.groupby("scheme", sort=False):
        last = grp.tail(1).reset_index(drop=True)
        V_T = np.zeros((len(res_pairs),), dtype=float)
        for idx, (_, col) in enumerate(res_pairs):
            V_T[idx] = float(pd.to_numeric(last[col], errors="coerce").iloc[0])

        V_T_tensor = torch.tensor(V_T, dtype=torch.float32)
        V_low_excess = (V_lo_ref - V_T_tensor).clamp_min(0.0)
        V_high_excess = (V_T_tensor - V_hi_ref).clamp_min(0.0)
        V_gap = V_low_excess + V_high_excess
        H_gap_list: List[float]

        if curve is not None:
            H_T = curve.v2h_all(V_T_tensor)
            H_lo = H_lo_ref.to(H_T.dtype)
            H_hi = H_hi_ref.to(H_T.dtype)
            H_low_excess = (H_lo - H_T).clamp_min(0.0)
            H_high_excess = (H_T - H_hi).clamp_min(0.0)
            H_gap_list = [float(v) for v in (H_low_excess + H_high_excess).tolist()]
        else:
            H_gap_list = [0.0] * len(res_pairs)

        diff = np.abs(V_target - V_T)
        ok = bool(np.all(diff <= tol))
        if ok:
            pass_count += 1
        results[str(scheme_name)] = {
            "V_T": V_T.tolist(),
            "V_target": V_target.tolist(),
            "abs_diff": diff.tolist(),
            "V_gap_1e8m3": [float(v) for v in V_gap.tolist()],
            "h_gap_m": H_gap_list,
            "pass": ok,
        }

    return {
        "year": int(year),
        "reservoirs": [name for name, _ in res_pairs],
        "tolerance": float(tol),
        "total_schemes": int(len(results)),
        "pass_count": int(pass_count),
        "fail_count": int(len(results) - pass_count),
        "schemes": results,
    }


def evaluate_year(year: int, window_k: int, tol: float) -> Dict:
    need = diagnose_need_capacity(year, window_k=window_k)
    terminal = check_terminal_attainment(year, tol=tol)
    return {"year": int(year), "need_capacity": need, "terminal_attainment": terminal}


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified evaluation for generated schedules.")
    parser.add_argument("--years", nargs="+", type=int, required=True, help="Years to evaluate")
    parser.add_argument("--window", type=int, default=10, help="Terminal window K")
    parser.add_argument("--tol", type=float, default=1e-3, help="Terminal tolerance (亿m³)")
    args = parser.parse_args()

    for year in args.years:
        bundle = evaluate_year(int(year), window_k=int(args.window), tol=float(args.tol))
        out_path = RESULTS_DIR / f"evaluation_{year}.json"
        out_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[evaluate] wrote {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
