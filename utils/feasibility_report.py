#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build feasibility reports for generated schedules.

Outputs:
  - diverse_results/feasibility_{year}.json
  - diverse_results/feasibility_report.md
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

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
    from .evaluate_all import diagnose_need_capacity, check_terminal_attainment
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
    from evaluate_all import diagnose_need_capacity, check_terminal_attainment


OUT_DIR = RESULTS_DIR
YEARS = [2018, 2019, 2020, 2021, 2022]
T = 36
TOL = 1e-6

TOKENS = {
    "constraint_csv": "\u7ea6\u675f\u6761\u4ef6",  # 约束条件
    "storage": ["\u5e93\u5bb9", "storage", "volume"],
    "flow": ["\u6d41\u91cf", "flow"],
    "outflow": ["\u51fa\u5e93", "out"],
    "lower": ["\u4e0b\u9650", "min", "lower"],
    "upper": ["\u4e0a\u9650", "max", "upper"],
    "min": ["\u6700\u5c0f", "min"],
    "max": ["\u6700\u5927", "max"],
}


def _reservoirs_from_constraints() -> List[str]:
    constraints = load_constraints()
    names = constraints.get("system_config", {}).get("reservoirs", [])
    if isinstance(names, list) and names:
        return [str(x) for x in names]
    return [str(x) for x in DEFAULT_RESERVOIRS]


def _resolve_constraints_csv() -> Path:
    candidates = [
        BASE_DIR / "shuxing" / f"{TOKENS['constraint_csv']}.csv",
        BASE_DIR / "shuxing" / "constraints" / f"{TOKENS['constraint_csv']}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    for path in (BASE_DIR / "shuxing").glob("*.csv"):
        if TOKENS["constraint_csv"] in path.stem:
            return path
    raise FileNotFoundError(f"cannot locate constraints csv ({TOKENS['constraint_csv']}.csv)")


def _pick_cols(df: pd.DataFrame, name: str) -> Tuple[str, str, str, str]:
    vmin = find_column(df, [name, TOKENS["storage"], TOKENS["lower"]])
    vmax = find_column(df, [name, TOKENS["storage"], TOKENS["upper"]])
    qmin = find_column(df, [name, TOKENS["min"], TOKENS["outflow"], TOKENS["flow"]])
    qmax = find_column(df, [name, TOKENS["max"], TOKENS["outflow"], TOKENS["flow"]])
    return vmin, vmax, qmin, qmax


def _schedule_cols(df: pd.DataFrame, name: str) -> Tuple[str, str]:
    storage_col = find_column(df, [name, TOKENS["storage"]])
    flow_col = find_column(df, [name, TOKENS["outflow"], TOKENS["flow"]])
    return storage_col, flow_col


def _load_targets(reservoir_count: int) -> np.ndarray:
    cfg = load_config()
    arr = np.array(cfg.get("constraints", {}).get("target_storage", [0] * reservoir_count), dtype=float)
    if arr.size < reservoir_count:
        raise ValueError("constraints.target_storage length is insufficient")
    return arr[:reservoir_count].astype(float)


def _select_primary_scheme(df: pd.DataFrame) -> pd.DataFrame:
    if "scheme" not in df.columns:
        return df.copy()
    scheme_name = "scheme_01"
    if not np.any(df["scheme"].astype(str).eq(scheme_name)):
        scheme_name = str(df["scheme"].astype(str).iloc[0])
    return df[df["scheme"].astype(str) == scheme_name].copy()


def _calc_year(year: int) -> Dict:
    # CSV first to avoid stale xlsx masking latest inference output.
    try:
        df_all = load_schedule_result(year, scheme=None, prefer_xlsx=False)
    except FileNotFoundError:
        df_all = load_schedule_result(year, scheme=None, prefer_xlsx=True)

    cons_csv = _resolve_constraints_csv()
    cons = pd.read_csv(cons_csv, encoding="utf-8-sig")

    reservoirs = _reservoirs_from_constraints()
    targets = _load_targets(len(reservoirs))
    df1 = _select_primary_scheme(df_all).head(T).reset_index(drop=True)

    per_res: Dict[str, Dict[str, float]] = {}
    flow_viol_any = 0
    stor_viol_any = 0

    for idx, name in enumerate(reservoirs):
        try:
            v_col, q_col = _schedule_cols(df1, name)
            vmin_col, vmax_col, qmin_col, qmax_col = _pick_cols(cons, name)
        except KeyError:
            continue

        v = pd.to_numeric(df1[v_col], errors="coerce").to_numpy(dtype=float)
        q = pd.to_numeric(df1[q_col], errors="coerce").to_numpy(dtype=float)
        vmin = pd.to_numeric(cons[vmin_col], errors="coerce").to_numpy(dtype=float)[:T]
        vmax = pd.to_numeric(cons[vmax_col], errors="coerce").to_numpy(dtype=float)[:T]
        qmin = pd.to_numeric(cons[qmin_col], errors="coerce").to_numpy(dtype=float)[:T]
        qmax = pd.to_numeric(cons[qmax_col], errors="coerce").to_numpy(dtype=float)[:T]

        viol_flow_lo = int(np.sum(q < qmin - TOL))
        viol_flow_hi = int(np.sum(q > qmax + TOL))
        viol_stor_lo = int(np.sum(v < vmin - TOL))
        viol_stor_hi = int(np.sum(v > vmax + TOL))
        flow_viol_any += (viol_flow_lo + viol_flow_hi)
        stor_viol_any += (viol_stor_lo + viol_stor_hi)

        near_lo = float(np.mean(np.isclose(v, vmin, atol=1e-4)))
        near_hi = float(np.mean(np.isclose(v, vmax, atol=1e-4)))
        term_dev = float(abs(v[-1] - targets[idx])) if len(v) else float("nan")

        per_res[name] = {
            "flow_viol_low": viol_flow_lo,
            "flow_viol_high": viol_flow_hi,
            "storage_viol_low": viol_stor_lo,
            "storage_viol_high": viol_stor_hi,
            "bound_touch_lo_rate": near_lo,
            "bound_touch_hi_rate": near_hi,
            "terminal_deviation": term_dev,
        }

    stats_path = OUT_DIR / f"diversity_stats_{year}.json"
    extra: Dict = {}
    if stats_path.exists():
        try:
            extra = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception:
            extra = {}

    constraints = load_constraints()
    cfg = load_config()
    need_cap = diagnose_need_capacity(year, schedule_df=df_all.copy(), constraints=constraints, config=cfg)
    terminal = check_terminal_attainment(year, tol=TOL, schedule_df=df_all.copy(), config=cfg)
    terminal_reachability = extra.get("terminal_reachability", {}) if isinstance(extra, dict) else {}

    return {
        "year": int(year),
        "per_reservoir": per_res,
        "any_flow_violations": bool(flow_viol_any > 0),
        "any_storage_violations": bool(stor_viol_any > 0),
        "summary": {
            "flow_violations_total": int(flow_viol_any),
            "storage_violations_total": int(stor_viol_any),
        },
        "diversity_stats": extra,
        "terminal_reachability": terminal_reachability,
        "need_capacity": need_cap,
        "terminal_attainment": terminal,
    }


def _build_markdown(reports: Sequence[Dict]) -> str:
    lines: List[str] = []
    lines.append("# Feasibility Report (primary scheme)")
    lines.append("")
    lines.append("Legend:")
    lines.append("- terminal_dev: terminal storage deviation in 1e8 m3")
    lines.append("- touch_lo/touch_hi: fraction of periods touching storage lower/upper bounds")
    lines.append("")
    for rep in reports:
        year = rep["year"]
        summary = rep.get("summary", {})
        ds = rep.get("diversity_stats", {})
        need_cap = rep.get("need_capacity", {})
        terminal = rep.get("terminal_attainment", {})
        reach = rep.get("terminal_reachability", {})
        lines.append(f"## {year}")
        lines.append(
            f"- flow_constraint_violated: {bool(rep['any_flow_violations'])} "
            f"(count={summary.get('flow_violations_total', 0)})"
        )
        lines.append(
            f"- storage_constraint_violated: {bool(rep['any_storage_violations'])} "
            f"(count={summary.get('storage_violations_total', 0)})"
        )
        if ds:
            lines.append(
                "- terminal_error_mean={:.6f}; precheck_gap_max/avg={:.6f}/{:.6f}".format(
                    float(ds.get("terminal_error", 0.0)),
                    float(ds.get("precheck_gap_max", 0.0)),
                    float(ds.get("precheck_gap_avg", 0.0)),
                )
            )
            if "post_projection_hard_gap_max" in ds:
                lines.append(
                    "- post_projection_hard_gap_max={:.6f}; fallback_enabled={}".format(
                        float(ds.get("post_projection_hard_gap_max", 0.0)),
                        bool(ds.get("post_projection_fallback_enabled", False)),
                    )
                )
        if reach:
            lines.append(
                "- reachability_ratio={:.2%}; reachability_gap_max/avg={:.6f}/{:.6f}".format(
                    float(reach.get("reachable_ratio", 0.0)),
                    float(reach.get("max_gap", 0.0)),
                    float(reach.get("avg_gap", 0.0)),
                )
            )
            unreachable = reach.get("unreachable_reservoirs", []) or []
            if unreachable:
                lines.append("- unreachable_reservoirs: {}".format(", ".join(str(x) for x in unreachable)))
        if need_cap:
            gaps = [info["cap_minus_need"] for info in need_cap.get("per_reservoir", {}).values()]
            if gaps:
                lines.append("- min(capacity_minus_need)={:.3f} (1e8 m3)".format(min(gaps)))
        if terminal:
            lines.append(
                "- terminal_pass_rate: {}/{} (tol={:.6f} 1e8m3)".format(
                    terminal.get("pass_count", 0),
                    terminal.get("total_schemes", 0),
                    terminal.get("tolerance", TOL),
                )
            )
        for name, pr in rep.get("per_reservoir", {}).items():
            lines.append(
                "  - {}: terminal_dev={:.6f}e8m3, touch_lo={:.2%}, touch_hi={:.2%}".format(
                    name,
                    float(pr.get("terminal_deviation", 0.0)),
                    float(pr.get("bound_touch_lo_rate", 0.0)),
                    float(pr.get("bound_touch_hi_rate", 0.0)),
                )
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reports: List[Dict] = []
    for year in YEARS:
        try:
            rep = _calc_year(year)
            reports.append(rep)
            out_json = OUT_DIR / f"feasibility_{year}.json"
            out_json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[feasible] wrote {out_json.name}")
        except Exception as exc:
            print(f"[feasible] skip {year}: {exc}")

    out_md = OUT_DIR / "feasibility_report.md"
    out_md.write_text(_build_markdown(reports), encoding="utf-8")
    print(f"[feasible] wrote {out_md}")


if __name__ == "__main__":  # pragma: no cover
    main()
