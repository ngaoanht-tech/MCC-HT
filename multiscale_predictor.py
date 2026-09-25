#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multiscale transformer training script for annual reservoir scheduling.

This module rebuilds the previously corrupted training pipeline.  It
loads annual datasets, trains the hierarchical transformer with optional
constraint and terminal-loss terms, evaluates results, and can generate
diverse annual schedules using Monte-Carlo dropout sampling.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import os
import sys
from pathlib import Path
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass
from dataclasses import dataclass, field
import copy
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from constraint.strict_converter import StrictWaterLevelConverter
from constraint.differentiable_joint_projection import project_joint_schedule
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import joblib
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
import time

from data import AnnualReservoirDataset
from losses import (
    CompositeSchedulingLoss,
    InteriorBarrierLoss,
    PowerOptimizedLoss,
    RampLoss,
    StoragePathConstraintLoss,
    TerminalWindowLoss,
    WeightedMSELoss,
    calculate_performance_metrics,
    PowerEnergySurrogate,
    TorchCurves,
)
from constraint.limits import make_level_limits
from constraint.projection_core import (
    q_box_from_level_step,
    soft_clip_projection,
    reachability_shrink,
)
from feature_engineering import build_multiscale_features
from models import HierarchicalTransformerPredictor, MultiScaleTCN

# Inline hydrologic helpers so the transformer package is self-contained.
_SCRIPT_DIR = Path(__file__).resolve().parent
_SHUXING_DIR = _SCRIPT_DIR / "shuxing"
_CONSTRAINT_PATH = _SCRIPT_DIR / "constraint" / "constraints_config.json"
_DEFAULT_RESERVOIRS = (
    "\u4e4c\u4e1c\u5fb7",
    "\u767d\u9e64\u6ee9",
    "\u6eaa\u6d1b\u6e21",
    "\u5411\u5bb6\u575d",
    "\u4e09\u5ce1",
    "\u845b\u6d32\u575d",
)


@lru_cache(maxsize=1)
def _load_constraint_config() -> Dict[str, Any]:
    from exceptions import ConfigurationError
    
    if not _CONSTRAINT_PATH.exists():
        raise ConfigurationError(
            f"约束配置文件不存在: {_CONSTRAINT_PATH}\n"
            f"请检查 constraint/constraints_config.json 文件是否存在。"
        )
    
    try:
        with open(_CONSTRAINT_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigurationError(
            f"约束配置文件格式错误: {_CONSTRAINT_PATH}\n"
            f"JSON 解析失败: {e}"
        )
    except Exception as e:
        raise ConfigurationError(
            f"无法读取约束配置文件 {_CONSTRAINT_PATH}: {e}"
        )
    
    return config


_CONSTRAINT_CFG = _load_constraint_config()
if _CONSTRAINT_CFG:
    _SYSTEM_CFG = _CONSTRAINT_CFG.get("system_config", {})
    _RESERVOIR_NAMES = tuple(_SYSTEM_CFG.get("reservoirs", _DEFAULT_RESERVOIRS))
    initial_levels_cfg = (
        _CONSTRAINT_CFG.get("water_balance_config", {})
        .get("initial_water_levels", {})
        .get("default_levels", [0.0] * len(_RESERVOIR_NAMES))
    )
    DEFAULT_INITIAL_LEVELS = {
        name: float(level) for name, level in zip(_RESERVOIR_NAMES, initial_levels_cfg)
    }
    hmax_cfg = _CONSTRAINT_CFG.get("water_level_constraints", {}).get("Hmax", [])
    DEFAULT_FINAL_LEVELS = {
        name: float(levels[-1]) if levels else float(DEFAULT_INITIAL_LEVELS.get(name, 0.0))
        for name, levels in zip(_RESERVOIR_NAMES, hmax_cfg)
    }
    time_steps = _CONSTRAINT_CFG.get("water_balance_config", {}).get("time_step_days", [])
    seconds_per_day = float(_CONSTRAINT_CFG.get("water_balance_config", {}).get("seconds_per_day", 86_400))
    _TIME_STEP_SECONDS = [float(day) * seconds_per_day for day in time_steps] or [10 * 24 * 3600] * 36
    CFG_TERMINAL_WINDOW = 36
    CFG_TERMINAL_TOL = 0.05
    CFG_TERMINAL_MAX_ITER = 200
    _HAVE_PSO_CONFIG = True
else:
    _RESERVOIR_NAMES = _DEFAULT_RESERVOIRS
    DEFAULT_INITIAL_LEVELS = {}
    DEFAULT_FINAL_LEVELS = {}
    _TIME_STEP_SECONDS = [10 * 24 * 3600] * 36
    CFG_TERMINAL_WINDOW = 36
    CFG_TERMINAL_TOL = 0.05
    CFG_TERMINAL_MAX_ITER = 200
    _HAVE_PSO_CONFIG = False

# Self-consistent inflow projection settings
# Safeguard defaults; can be overridden by config.yaml if present
SELF_CONSISTENT_ITERS_DEFAULT = 2
SELF_CONSISTENT_TOL = 1e-4


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required csv: {path}")
    last_exc: Optional[Exception] = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Failed to read csv {path}: {last_exc}")



def load_storage_constraints_from_csv(
    csv_path: Path,
    reservoir_names: Sequence[str],
    periods: int = 36,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    df = pd.read_csv(str(csv_path), encoding="utf-8-sig")
    cols = [str(c).strip() for c in df.columns]

    def _pick_storage_col(res_name: str, keywords: Sequence[str]) -> str:
        for keyword in keywords:
            for col in cols:
                col_norm = str(col).strip()
                if res_name in col_norm and keyword in col_norm:
                    return col_norm
        for col in cols:
            col_norm = str(col).strip()
            lowered = col_norm.lower()
            if res_name in col_norm and any(token in lowered for token in ("min", "max", "lower", "upper")):
                return col_norm
        raise KeyError(f"missing storage column for {res_name} ({keywords[0]})")

    keyword_down = "库容下限"
    keyword_up = "库容上限"

    vmin_cols = []
    vmax_cols = []
    for name in reservoir_names:
        vmin_cols.append(_pick_storage_col(name, (keyword_down, "下限", "lower")))
        vmax_cols.append(_pick_storage_col(name, (keyword_up, "上限", "upper")))

    Vmin_np = df[vmin_cols].iloc[:periods].to_numpy(dtype=float)
    Vmax_np = df[vmax_cols].iloc[:periods].to_numpy(dtype=float)
    Vmin_np = np.minimum(Vmin_np, Vmax_np)

    first_idx = 0
    last_idx = min(len(Vmin_np) - 1, periods - 1)
    V0_np = ((Vmin_np[first_idx] + Vmax_np[first_idx]) * 0.5).astype(np.float32)
    VT_np = ((Vmin_np[last_idx] + Vmax_np[last_idx]) * 0.5).astype(np.float32)

    dev = device or torch.device("cpu")
    Vmin = torch.tensor(Vmin_np[:periods], dtype=torch.float32, device=dev)
    Vmax = torch.tensor(Vmax_np[:periods], dtype=torch.float32, device=dev)
    V0 = torch.tensor(V0_np, dtype=torch.float32, device=dev).unsqueeze(0)
    VT = torch.tensor(VT_np, dtype=torch.float32, device=dev).unsqueeze(0)
    return Vmin, Vmax, V0, VT


def compute_initial_terminal_storage_from_Hmax(
    csv_path: Path,
    reservoir_names: Sequence[str],
    converter: StrictWaterLevelConverter,
    periods: int = 36,
    tol: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute V0/VT from 水位上限 via curves; enforce consistency.

    - V0: use period-1 水位上限 mapped to storage via curves
    - VT: use period-36 水位上限 mapped to storage via curves
    - If not nearly equal (max diff > tol), set both to VT (ensure 一?
    Returns tensors shaped [1, R] in 亿m³.
    """
    df = pd.read_csv(str(csv_path), encoding="utf-8-sig")
    if len(df) < periods:
        raise ValueError(f"约束文件 {csv_path} 时段不足 {periods}")
    hmax_cols = [f"{name}水位上限" for name in reservoir_names]
    missing = [c for c in hmax_cols if c not in df.columns]
    if missing:
        raise KeyError(f"constraints table missing Hmax columns: {missing}")
    H0 = torch.tensor(df[hmax_cols].iloc[0].to_numpy(dtype=float), dtype=torch.float32)
    HT = torch.tensor(df[hmax_cols].iloc[periods - 1].to_numpy(dtype=float), dtype=torch.float32)
    V0 = converter.to_volume(H0).squeeze(0)  # [R]
    VT = converter.to_volume(HT).squeeze(0)  # [R]
    if torch.max(torch.abs(V0 - VT)).item() > tol:
        # Enforce consistency: both equal to terminal-based storage
        V0 = VT.clone()
    return V0.unsqueeze(0), VT.unsqueeze(0)


def load_static_params(reservoir_names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    station_csv_candidates = [
        _SHUXING_DIR / "电站特性.csv",
        _SHUXING_DIR / "金沙江四库特征参数.csv",
    ]
    station_path = next((p for p in station_csv_candidates if p.exists()), None)
    if station_path is None:
        raise FileNotFoundError(
            f"missing station characteristics csv, tried: {[str(p) for p in station_csv_candidates]}"
        )
    station_chars = _read_csv(station_path)
    station_chars.columns = station_chars.columns.astype(str).str.strip()

    name_col = None
    for cand in ("水库名称", "电站名称", "水库", "电站"):
        if cand in station_chars.columns:
            name_col = cand
            break
    if name_col is None:
        # Fallback to first text-like column.
        for col in station_chars.columns:
            if station_chars[col].dtype == object:
                name_col = col
                break
    if name_col is None:
        raise ValueError(f"{station_path} missing reservoir-name column")

    cap_col = None
    for cand in ("装机容量", "装机容量(MW)", "总装机容量", "总装机", "装机规模"):
        if cand in station_chars.columns:
            cap_col = cand
            break
    if cap_col is None:
        cap_col = station_chars.columns[0]

    station_chars[name_col] = station_chars[name_col].astype(str).str.strip()

    static_params: Dict[str, Dict[str, Any]] = {}
    for name in reservoir_names:
        row = station_chars[station_chars[name_col] == name]
        if row.empty:
            row = station_chars[station_chars[name_col].str.contains(str(name), na=False)]
        if row.empty:
            raise ValueError(f"reservoir '{name}' not found in station table {station_path}")
        row = row.iloc[0]
        capacity = float(row.get(cap_col, 0.0) or 0.0)
        p_min = capacity * 0.15 if capacity > 0 else 0.0

        z_v_candidates = [
            _SHUXING_DIR / "curves" / f"{name}.csv",
            _SHUXING_DIR / f"{name}水位-库容曲线.csv",
        ]
        z_v_path = next((p for p in z_v_candidates if p.exists()), None)
        if z_v_path is None:
            raise FileNotFoundError(f"missing level-storage curve for {name}, tried: {[str(p) for p in z_v_candidates]}")
        z_v_curve = _read_csv(z_v_path)

        q_tail_candidates = [
            _SHUXING_DIR / f"{name}出库流量-下游水位曲线.csv",
            _SHUXING_DIR / f"{name}流量-下游水位曲线.csv",
        ]
        q_tail_path = next((p for p in q_tail_candidates if p.exists()), None)
        if q_tail_path is None:
            raise FileNotFoundError(f"missing tailwater curve for {name}, tried: {[str(p) for p in q_tail_candidates]}")
        q_tail_curve = _read_csv(q_tail_path)

        static_params[name] = {
            "capacity_installed": capacity,
            "p_out_min": p_min,
            "z_v_curve": z_v_curve,
            "q_z_tail_curve": q_tail_curve,
        }
    return static_params


def load_and_convert_bounds(
    reservoirs: Sequence[str],
    *,
    constraints_dir: str,
    curves_dir: str,
    volume_scale: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StrictWaterLevelConverter]:
    """Load per-period H bounds and convert to storage bounds via strict curves."""

    T = 36
    R = len(reservoirs)
    Hmin = np.zeros((T, R), dtype=np.float32)
    Hmax = np.zeros((T, R), dtype=np.float32)

    if not os.path.isabs(constraints_dir):
        constraints_dir = os.path.join(_SCRIPT_DIR, constraints_dir)
    if not os.path.isabs(curves_dir):
        curves_dir = os.path.join(_SCRIPT_DIR, curves_dir)

    if os.path.isdir(constraints_dir):
        import pandas as pd  # local import to avoid circular dependency

        for idx, name in enumerate(reservoirs):
            path = os.path.join(constraints_dir, f"{name}.csv")
            if not os.path.exists(path):
                raise FileNotFoundError(f"缺少约束文件: {path}")
            df = pd.read_csv(path, encoding="utf-8-sig")

            def _find_col(keys: List[str]) -> str:
                for key in keys:
                    for col in df.columns:
                        if key in str(col):
                            return col
                raise KeyError(f"{name} 约束文件缺少列，候?{keys}，实?{list(df.columns)}")

            hmin_col = _find_col(["H_min", "ˮλС", "Сˮλ"])
            hmax_col = _find_col(["H_max", "ˮλ", "ˮλ"])
            Hmin[:, idx] = df[hmin_col].iloc[:T].astype(float).to_numpy()
            Hmax[:, idx] = df[hmax_col].iloc[:T].astype(float).to_numpy()
    else:
        wl_cfg = _CONSTRAINT_CFG.get("water_level_constraints", {})
        Hmin_cfg = wl_cfg.get("Hmin")
        Hmax_cfg = wl_cfg.get("Hmax")
        if Hmin_cfg is None or Hmax_cfg is None:
            raise RuntimeError("constraints_config.json 缺少 water_level_constraints.Hmin/Hmax")
        Hmin = np.asarray(Hmin_cfg, dtype=np.float32).T  # [T,R]
        Hmax = np.asarray(Hmax_cfg, dtype=np.float32).T

    converter = StrictWaterLevelConverter(list(reservoirs), curves_dir=curves_dir, volume_scale=volume_scale).to(device)

    Hmin_t = torch.tensor(Hmin, device=device)
    Hmax_t = torch.tensor(Hmax, device=device)
    Vmin = converter.to_volume(Hmin_t)
    Vmax = converter.to_volume(Hmax_t)

    if get_config is None:
        raise RuntimeError("get_config() missing; cannot read initial/target levels")
    cfg = get_config()
    H0_cfg = cfg.get("constraints.initial_levels")
    HT_cfg = cfg.get("constraints.target_levels")
    if not H0_cfg or not HT_cfg:
        raise ValueError("config.yaml constraints.initial_levels/target_levels 必须提供")
    if len(H0_cfg) != R or len(HT_cfg) != R:
        raise ValueError("constraints.initial_levels/target_levels 长度必须等于水库数量")

    H0 = torch.tensor(H0_cfg, dtype=torch.float32, device=device)
    HT = torch.tensor(HT_cfg, dtype=torch.float32, device=device)
    V0 = converter.to_volume(H0).unsqueeze(0)
    VT = converter.to_volume(HT).unsqueeze(0)

    return Vmin, Vmax, V0, VT, converter


def _to_float_array(values: Iterable[Iterable[float]], periods: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError("constraint arrays must be 2-D")
    return arr[:, :periods]


def _interval_column(up: str, dn: str) -> str:
    return f"{up}-{dn}_区间来水(m3/s)"


def _interval_column_candidates(up: str, dn: str) -> List[str]:
    """Return common interval inflow column spellings used by the CSV files."""

    base = f"{up}-{dn}"
    return [
        f"{base}区间来水(m3/s)",
        f"{base}_区间来水(m3/s)",
        f"{base}区间来水(m³/s)",
        f"{base}_区间来水(m³/s)",
    ]


def _find_inflow_csv(year: Optional[int]) -> Path:
    candidates: List[Path] = []
    if year is not None:
        year = int(year)
        candidates.extend(
            [
                _SCRIPT_DIR / folder / f"{year}.csv"
                for folder in ("test", "train", "validation")
            ]
        )
    else:
        candidates.extend(sorted((_SCRIPT_DIR / "test").glob("*.csv")))
        candidates.extend(sorted((_SCRIPT_DIR / "train").glob("*.csv")))
        candidates.extend(sorted((_SCRIPT_DIR / "validation").glob("*.csv")))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("no inflow csv found under test/train/validation")


def load_dynamic_constraints_and_inflows(
    reservoir_names: Sequence[str],
    num_periods: int = 36,
    inflow_year: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    if not _CONSTRAINT_CFG:
        raise RuntimeError("constraint config is empty; cannot load dynamic constraints")

    num_periods = int(num_periods)
    config_order = list(_RESERVOIR_NAMES)
    indices = []
    for name in reservoir_names:
        if name not in config_order:
            raise ValueError(f"reservoir '{name}' is not defined in constraint config")
        indices.append(config_order.index(name))

    flow_constraints = _CONSTRAINT_CFG["flow_constraints"]
    level_constraints = _CONSTRAINT_CFG["water_level_constraints"]

    qmin = _to_float_array(flow_constraints["Qmin"], num_periods)
    qmax = _to_float_array(flow_constraints["Qmax"], num_periods)
    hmin = _to_float_array(level_constraints["Hmin"], num_periods)
    hmax = _to_float_array(level_constraints["Hmax"], num_periods)
    dh_down = level_constraints.get("dH_down")
    dh_up = level_constraints.get("dH_up")
    dH_down_arr = _to_float_array(dh_down, num_periods) if dh_down else None
    dH_up_arr = _to_float_array(dh_up, num_periods) if dh_up else None

    dyn_constraints: Dict[str, pd.DataFrame] = {}
    for name, idx in zip(reservoir_names, indices):
        if dH_down_arr is not None:
            span_down = np.clip(dH_down_arr[idx, :num_periods], 0.5, None)
        else:
            span_down = np.clip(hmax[idx, :num_periods] - hmin[idx, :num_periods], 0.5, None)
        if dH_up_arr is not None:
            span_up = np.clip(dH_up_arr[idx, :num_periods], 0.5, None)
        else:
            span_up = span_down
        data = {
            "level_min": hmin[idx, :num_periods],
            "level_max": hmax[idx, :num_periods],
            "q_out_min": qmin[idx, :num_periods],
            "q_out_max": qmax[idx, :num_periods],
            "level_down_max": span_down,
            "level_up_max": span_up,
        }
        dyn_constraints[name] = pd.DataFrame(data).reset_index(drop=True)

    inflow_csv = _find_inflow_csv(inflow_year)
    df_inflow = _read_csv(inflow_csv).head(num_periods).copy()
    df_inflow.columns = df_inflow.columns.str.strip()
    # Normalize unicode superscript ³ to ASCII 3 to match expected column names
    df_inflow.columns = df_inflow.columns.str.replace("\u00b3", "3")
    if "Ѯ" in df_inflow.columns:
        df_inflow = df_inflow.drop(columns=["Ѯ"])

    head_col = f"{reservoir_names[0]}_入库流量(m3/s)"
    if head_col not in df_inflow.columns:
        raise ValueError(f"{inflow_csv} missing required inflow column: {head_col}")
    inflow_data = pd.DataFrame({"inflow_head": df_inflow[head_col].astype(float).to_numpy()})

    interval_dict: Dict[str, np.ndarray] = {}
    for up, dn in zip(reservoir_names[:-1], reservoir_names[1:]):
        key = f"interval_{up}_{dn}"
        series = None
        for col_name in _interval_column_candidates(up, dn):
            if col_name in df_inflow.columns:
                series = df_inflow[col_name]
                break
        interval_dict[key] = series.astype(float).to_numpy() if series is not None else np.zeros(len(inflow_data))
    interval_inflows = pd.DataFrame(interval_dict).head(num_periods).reset_index(drop=True)

    return inflow_data, interval_inflows, dyn_constraints


def _derive_reservoir_names_from_data(year: int) -> Optional[List[str]]:
    """Fallback: return configured default reservoirs.

    The original impl was corrupted by non-UTF8 literals; keep logic simple and safe.
    """
    try:
        return list(_DEFAULT_RESERVOIRS)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Minimal training stub and checkpoint discovery for generation-only runs
# ---------------------------------------------------------------------------
def _discover_multiscale_checkpoint(results_dir: Path) -> Optional[Tuple[Path, Path, Dict[str, Any]]]:
    """Find a *_best_model.pth with a matching *_hyperparameters.json that indicates multiscale."""
    if not results_dir.exists():
        return None
    candidates: List[Tuple[float, Path, Path, Dict[str, Any]]] = []
    for model_file in results_dir.glob("*_best_model.pth"):
        name = model_file.stem.replace("_best_model", "")
        params_file = results_dir / f"{name}_hyperparameters.json"
        if not params_file.exists():
            continue
        try:
            with open(params_file, "r", encoding="utf-8-sig") as f:
                params = json.load(f)
            mt = params.get("model_type", "transformer")
            if mt == "multiscale":
                mtime = model_file.stat().st_mtime
                candidates.append((mtime, model_file, params_file, params))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, model_file, params_file, params = candidates[0]
    return model_file, params_file, params


def _load_multiscale_model(model_path: Path, params: Dict[str, Any]) -> torch.nn.Module:
    """Build HierarchicalTransformerPredictor and load weights (robust to partial mismatches)."""
    input_dim = int(params.get("input_dim", 22))
    output_dim = int(params.get("output_dim", 6))
    sequence_length = int(params.get("sequence_length", 36))
    dropout = float(params.get("dropout", 0.1))
    output_sequence_length = int(params.get("output_sequence_length", output_dim))
    enable_multiscale = bool(params.get("enable_multiscale", True))
    enable_sequence_decoding = bool(params.get("enable_sequence_decoding", False))

    model = HierarchicalTransformerPredictor(
        input_dim=input_dim,
        output_dim=output_dim,
        sequence_length=sequence_length,
        dropout=dropout,
        output_sequence_length=output_sequence_length,
        enable_multiscale=enable_multiscale,
        enable_sequence_decoding=enable_sequence_decoding,
    )
    ckpt = torch.load(str(model_path), map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
        if "multiscale_tcn_state_dict" in ckpt:
            setattr(model, "multiscale_tcn_state_dict", ckpt["multiscale_tcn_state_dict"])
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Level-informed projection helper (used during generation)
# ---------------------------------------------------------------------------
def _apply_level_projection(
    q_raw: torch.Tensor,
    q_in: torch.Tensor,
    V0: torch.Tensor,
    dt_seconds: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    level_limits: Dict[str, torch.Tensor],
    curves: TorchCurves,
    V_T_lo: Optional[torch.Tensor] = None,
    V_T_hi: Optional[torch.Tensor] = None,
    reach_cfg: Optional[Dict[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project flows inside the level/ramp-induced box while respecting [q_min, q_max] and optional drop limits.

    Returns
    -------
    (q_proj, q_level_min, q_level_max)
        q_proj : projected flows [B,T,R]
        q_level_min/q_level_max : level-induced bounds used during projection.
    """
    if q_raw is None or q_raw.numel() == 0:
        return q_raw, torch.empty_like(q_raw), torch.empty_like(q_raw)

    if q_raw.dim() != 3:
        raise ValueError("q_raw must have shape [B,T,R]")

    device = q_raw.device
    dtype = q_raw.dtype
    q_proj = q_raw.clone()
    q_in = q_in.to(device=device, dtype=dtype)
    q_min = q_min.to(device=device, dtype=dtype)
    q_max = q_max.to(device=device, dtype=dtype)

    B, T, R = q_proj.shape

    curves = curves.to(device)
    curves.eval()

    H_min = level_limits["H_min"].to(device=device, dtype=dtype)
    H_max = level_limits["H_max"].to(device=device, dtype=dtype)
    dH_up = level_limits["dH_up"].to(device=device, dtype=dtype)
    dH_dn = level_limits["dH_dn"].to(device=device, dtype=dtype)

    if H_min.dim() == 2:
        H_min = H_min.unsqueeze(0)
    if H_max.dim() == 2:
        H_max = H_max.unsqueeze(0)
    if dH_up.dim() == 2:
        dH_up = dH_up.unsqueeze(0)
    if dH_dn.dim() == 2:
        dH_dn = dH_dn.unsqueeze(0)

    if H_min.size(0) == 1 and H_min.size(0) != B:
        H_min = H_min.expand(B, -1, -1)
    if H_max.size(0) == 1 and H_max.size(0) != B:
        H_max = H_max.expand(B, -1, -1)
    if dH_up.size(0) == 1 and dH_up.size(0) != B:
        dH_up = dH_up.expand(B, -1, -1)
    if dH_dn.size(0) == 1 and dH_dn.size(0) != B:
        dH_dn = dH_dn.expand(B, -1, -1)

    dt_seconds = dt_seconds.to(device=device, dtype=dtype)
    if dt_seconds.dim() == 1:
        dt_seconds = dt_seconds.view(1, -1, 1)
    elif dt_seconds.dim() == 2:
        dt_seconds = dt_seconds.unsqueeze(-1)
    elif dt_seconds.dim() != 3:
        raise ValueError("dt_seconds must have shape [T], [1,T] or [1,T,1]")
    dt_vol = (dt_seconds / 1e8).expand(B, -1, R)

    V_t = V0.to(device=device, dtype=dtype)
    if V_t.dim() == 1:
        V_t = V_t.unsqueeze(0).expand(B, -1)
    elif V_t.size(0) == 1 and V_t.size(0) != B:
        V_t = V_t.expand(B, -1)

    level_qmin = torch.zeros_like(q_proj)
    level_qmax = torch.zeros_like(q_proj)
    prev_q: Optional[torch.Tensor] = None

    use_reach = False
    alpha = 0.6
    if isinstance(reach_cfg, dict):
        try:
            use_reach = bool(reach_cfg.get("enabled", False)) and (V_T_lo is not None) and (V_T_hi is not None)
            alpha = float(reach_cfg.get("shrink_alpha", 0.6))
        except Exception:
            use_reach = False
            alpha = 0.6

    for t in range(T):
        H_t = curves.v2h_all(V_t)
        k_t = dt_vol[:, t, :].expand(B, R)
        Hmin_t = H_min[:, t, :]
        Hmax_t = H_max[:, t, :]
        dH_up_t = dH_up[:, t, :]
        dH_dn_t = dH_dn[:, t, :]

        qL_lvl, qU_lvl = q_box_from_level_step(
            curves=curves,
            V_t=V_t,
            H_t=H_t,
            Qin_t=q_in[:, t, :],
            k_t=k_t,
            Hmin_t=Hmin_t,
            Hmax_t=Hmax_t,
            dH_up_t=dH_up_t,
            dH_dn_t=dH_dn_t,
        )
        # Optionally shrink by terminal reachability using remaining window
        qL_use, qU_use = qL_lvl, qU_lvl
        if use_reach and (t + 1 < T):
            try:
                qL_use, qU_use = reachability_shrink(
                    qL=qL_lvl,
                    qU=qU_lvl,
                    V_t=V_t,
                    Qin_future=q_in[:, t + 1 :, :],
                    qmin_future=q_min[:, t + 1 :, :],
                    qmax_future=q_max[:, t + 1 :, :],
                    k_future=dt_vol[:, t + 1 :, :],
                    V_T_lo=V_T_lo if V_T_lo is not None else V_t.new_zeros((B, R)),
                    V_T_hi=V_T_hi if V_T_hi is not None else V_t.new_zeros((B, R)),
                    alpha=alpha,
                )
            except Exception:
                qL_use, qU_use = qL_lvl, qU_lvl

        q_step = soft_clip_projection(
            q_raw=q_proj[:, t, :],
            q_min=q_min[:, t, :],
            q_max=q_max[:, t, :],
            q_box_lo=qL_use,
            q_box_hi=qU_use,
        )
        q_proj[:, t, :] = q_step
        prev_q = q_step
        level_qmin[:, t, :] = qL_lvl
        level_qmax[:, t, :] = qU_lvl
        V_t = V_t + (q_in[:, t, :] - q_step) * k_t

    # NOTE: terminal correction of flows was previously applied here by
    # redistributing V_T_target - V_final across all time steps based on slack.
    # This turned out to introduce large residual errors and instability.
    # For now we only rely on HierarchicalProjection + training-time losses
    # to enforce the terminal target and keep _apply_level_projection
    # as a pure level-feasibility helper without modifying q_proj further.

    return q_proj, level_qmin, level_qmax


def _joint_storage_bounds(
    V_min: torch.Tensor,
    V_max: torch.Tensor,
    level_limits: Optional[Dict[str, torch.Tensor]],
    curves: Optional[TorchCurves],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Include absolute water-level bounds; existing level-change code is unchanged."""
    if level_limits is None or curves is None:
        return V_min, V_max
    level_vmin = curves.h2v_all(level_limits["H_min"])
    level_vmax = curves.h2v_all(level_limits["H_max"])
    return torch.maximum(V_min, level_vmin), torch.minimum(V_max, level_vmax)


# ---------------------------------------------------------------------------
# Lightweight helpers for smoke tests and unified inference path
# ---------------------------------------------------------------------------
def build_dataloaders(
    split: str = "train",
    batch_size: Optional[int] = None,
    num_workers: int = 0,
    precomputed: Optional[Dict[str, Any]] = None,
) -> DataLoader:
    """Construct a DataLoader for the requested split using config defaults."""

    if not CONFIG_AVAILABLE or get_config is None:
        raise RuntimeError("Configuration loader is required to build dataloaders.")

    cfg_obj = get_config()

    def _cfg_get(obj: Any, key: str, default: Any) -> Any:
        if hasattr(obj, "get"):
            return obj.get(key, default)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    data_cfg = _cfg_get(cfg_obj, "data", {})
    years_cfg = data_cfg.get("years", {}) if isinstance(data_cfg, dict) else {}
    split_key = split.lower()
    if split_key in {"val", "validation"}:
        split_key = "validation"
    elif split_key in {"test", "testing"}:
        split_key = "test"
    else:
        split_key = "train"

    years = years_cfg.get(split_key, [])
    if not years:
        raise ValueError(f"No years configured for split '{split}'.")
    years = [int(y) for y in years]

    data_dir = data_cfg.get("train_data_dir", "train")
    if split_key == "test":
        data_dir = data_cfg.get("test_data_dir", data_dir if data_dir != "train" else "test")

    preprocess_cfg = data_cfg.get("preprocessing", {}) if isinstance(data_cfg, dict) else {}
    use_log_transform = bool(preprocess_cfg.get("use_log_transform", True))
    normalize = bool(preprocess_cfg.get("normalize", True))

    try:
        sequence_length = int(cfg_obj.get("model.sequence_length", 36))  # type: ignore[arg-type]
    except Exception:
        sequence_length = 36

    batch_sz = batch_size or 8
    fit_transforms = split_key == "train" and precomputed is None

    dataset = AnnualReservoirDataset(
        data_dir=data_dir,
        years=years,
        sequence_length=sequence_length,
        use_log_transform=use_log_transform,
        normalize=normalize,
        fit_transforms=fit_transforms,
        precomputed_transforms=precomputed,
    )
    return DataLoader(dataset, batch_size=batch_sz, shuffle=(split_key == "train"), num_workers=num_workers)


def build_model(sample_batch: Optional[Dict[str, torch.Tensor]] = None) -> HierarchicalTransformerPredictor:
    """Initialise the multiscale model using inferred input/output dimensions."""

    if sample_batch is None:
        loader = build_dataloaders(split="train", batch_size=1)
        sample_batch = next(iter(loader))

    features: torch.Tensor = sample_batch["features"]
    targets: torch.Tensor = sample_batch["targets"]

    input_dim = int(features.shape[-1])
    output_dim = int(targets.shape[-1])
    sequence_length = int(features.shape[-2])
    output_sequence_length = int(targets.shape[-2])

    return HierarchicalTransformerPredictor(
        input_dim=input_dim,
        output_dim=output_dim,
        sequence_length=sequence_length,
        output_sequence_length=output_sequence_length,
    )


def run_multiscale_inference(
    years: Optional[Sequence[int]] = None,
    samples: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Load the persisted checkpoint and decode schedules directly without repairs."""

    if samples is not None and samples < 1:
        raise ValueError("samples must be >= 1.")

    cfg_obj = get_config() if CONFIG_AVAILABLE and get_config is not None else {}

    def _cfg_get(obj: Any, key: str, default: Any) -> Any:
        if hasattr(obj, "get"):
            return obj.get(key, default)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    data_cfg = _cfg_get(cfg_obj, "data", {})
    years_cfg = data_cfg.get("years", {}) if isinstance(data_cfg, dict) else {}
    gen_cfg = cfg_obj.get("generation", {}) if cfg_obj else {}
    default_years = (gen_cfg.get("years") or years_cfg.get("test", []) or years_cfg.get("validation", []))
    target_years = [int(y) for y in (years or default_years)]

    if not target_years:
        raise ValueError("No inference years provided or configured.")

    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / _cfg_get(data_cfg, "results_dir", "results")
    found = _discover_multiscale_checkpoint(results_dir)
    if not found:
        raise FileNotFoundError("No multiscale checkpoint found in results directory.")
    model_path, _, params = found
    model = _load_multiscale_model(model_path, params)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    preprocess_cfg = data_cfg.get("preprocessing", {}) if isinstance(data_cfg, dict) else {}
    use_log_transform = bool(preprocess_cfg.get("use_log_transform", True))
    normalize = bool(preprocess_cfg.get("normalize", True))

    transforms_path = results_dir / "annual_transforms.pkl"
    precomputed = None
    if transforms_path.exists():
        try:
            precomputed = AnnualReservoirDataset.load_transforms(str(transforms_path))
        except Exception:
            precomputed = None

    if normalize and precomputed is None:
        raise FileNotFoundError(
            f"Missing transforms file for normalized inference: {transforms_path}. "
            "Please retrain or restore results/annual_transforms.pkl."
        )

    data_dir_test = data_cfg.get("test_data_dir") if isinstance(data_cfg, dict) else None
    if not data_dir_test:
        data_dir_test = "test"
    # Resolve generation defaults from YAML when CLI args omitted
    gen_cfg = cfg_obj.get("generation", {}) if cfg_obj else {}
    if samples is None:
        try:
            samples = int(gen_cfg.get("population_size", 1))
        except Exception:
            samples = 1
    if output_dir is None:
        output_dir = str(gen_cfg.get("output_dir", "diverse_results"))
    os.makedirs(output_dir, exist_ok=True)
    try:
        sequence_length = int(cfg_obj.get("model.sequence_length", 36))  # type: ignore[arg-type]
    except Exception:
        sequence_length = 36

    for year in target_years:
        dataset = AnnualReservoirDataset(
            data_dir=data_dir_test,
            years=[int(year)],
            sequence_length=sequence_length,
            use_log_transform=use_log_transform,
            normalize=normalize,
            fit_transforms=False,
            precomputed_transforms=precomputed,
        )
        if len(dataset) == 0:
            print(f"[infer] year={year}: dataset empty, skipping.")
            continue

        sample = dataset[-1]
        features = sample["features"].unsqueeze(0).to(device)
        _, stats = generate_diverse_annual_schedules(
            model=model,
            test_input=features,
            population_size=samples,
            year=int(year),
            output_dir=output_dir,
            device=device,
            dataset=dataset,
        )
        print(
            f"[infer] year={year} completed -> violation={stats['violation_rate']:.6f}, "
            f"terminal_err={stats['terminal_error']:.6f}, infeasible={stats['infeasible_ratio']:.4f}"
        )


def train_multiscale_model(
    best_params: Dict[str, Any],
    enable_constraints: bool = True,
    constraint_config_path: Optional[str] = None,
    enable_power_optimization: bool = True,
    power_weight: float = 0.1,
    flow_weight: float = 1.0,
    enable_mc_dropout: bool = True,
    mc_dropout_rate: float = 0.2,
    **_: Any,
) -> Tuple[torch.nn.Module, float, float, float, float]:
    """Train the multiscale transformer from scratch and persist the best checkpoint."""

    script_dir = Path(__file__).resolve().parent
    cfg = get_config() if CONFIG_AVAILABLE and get_config is not None else {}

    training_cfg_root = cfg.get("training", {}) if cfg else {}
    seed_value = training_cfg_root.get("seed", None) if isinstance(training_cfg_root, dict) else None
    env_seed = os.getenv("HYDRO_TRAIN_SEED")
    if env_seed not in (None, ""):
        seed_value = env_seed
    if seed_value not in (None, ""):
        try:
            _set_global_seed(int(seed_value))
            print(f"[seed] using seed={int(seed_value)}")
        except Exception as exc:
            print(f"[warn] failed to set global seed {seed_value}: {exc}")

    # ------------------------------------------------------------------
    # Train-only projection toggle (keeps inference projection intact)
    # ------------------------------------------------------------------
    # Used for ablation experiments: training without constraint projection,
    # but inference still applies projection to "repair" outputs.
    train_proj_cfg = (training_cfg_root.get("projection", {}) if isinstance(training_cfg_root, dict) else {}) or {}
    train_projection_enabled = bool(train_proj_cfg.get("enabled", True))

    data_cfg = cfg.get("data", {})
    year_cfg = data_cfg.get("years", {})
    train_years = (
        best_params.get("train_years")
        or year_cfg.get("train")
        or sorted(int(p.stem) for p in (script_dir / "train").glob("*.csv"))
    )
    val_years = (
        best_params.get("validation_years")
        or year_cfg.get("validation")
        or sorted(int(p.stem) for p in (script_dir / "train").glob("*.csv") if int(p.stem) not in train_years)[-5:]
    )
    if not train_years or not val_years:
        raise ValueError("training/validation years missing; provide in config.yaml or best_params")

    sequence_length = int(best_params.get("sequence_length", 36))
    output_sequence_length = int(best_params.get("output_sequence_length", sequence_length))
    dropout = float(best_params.get("dropout", mc_dropout_rate if enable_mc_dropout else 0.2))
    enable_sequence_decoding = bool(best_params.get("enable_sequence_decoding", True))
    enable_multiscale = bool(best_params.get("enable_multiscale", True))

    learning_rate = float(best_params.get("learning_rate", 1e-3))
    batch_size = int(best_params.get("batch_size", 8))
    training_cfg = cfg.get("training", {})
    max_epochs = int(best_params.get("max_epochs", training_cfg.get("max_epochs", 80)))
    if hasattr(cfg, "get"):
        trainer_override = cfg.get("trainer.max_epochs", None)
    elif isinstance(cfg, dict):
        trainer_override = cfg.get("trainer", {}).get("max_epochs")
    else:
        trainer_override = None
    if trainer_override not in (None, ""):
        try:
            max_epochs = int(trainer_override)
        except (TypeError, ValueError):
            pass
    patience = int(best_params.get("patience", training_cfg.get("patience", 10)))
    grad_clip = float(best_params.get("max_grad_norm", training_cfg.get("gradient_clipping", {}).get("max_norm", 1.0)))
    logging_cfg = training_cfg.get("logging", {}) if isinstance(training_cfg, dict) else {}
    log_mode = str(logging_cfg.get("mode", "standard")).strip().lower()
    if log_mode not in {"standard", "detailed"}:
        log_mode = "standard"
    loss_curve_cfg = logging_cfg.get("loss_curve", {}) if isinstance(logging_cfg, dict) else {}
    loss_curve_png = str(loss_curve_cfg.get("png", "training_loss_curve.png"))
    loss_curve_csv = str(loss_curve_cfg.get("csv", "training_loss_history.csv"))
    sc_iters_train = _resolve_self_consistent_iters(cfg, "training", SELF_CONSISTENT_ITERS_DEFAULT)
    sc_iters_val = _resolve_self_consistent_iters(cfg, "training", SELF_CONSISTENT_ITERS_DEFAULT)

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------
    transform_flags = data_cfg.get("preprocessing", {})
    use_log_transform = bool(best_params.get("use_log_transform", transform_flags.get("use_log_transform", True)))
    normalize = bool(best_params.get("normalize", transform_flags.get("normalize", True)))

    train_dataset = AnnualReservoirDataset(
        data_dir=data_cfg.get("train_data_dir", "train"),
        years=[int(y) for y in train_years],
        sequence_length=sequence_length,
        use_log_transform=use_log_transform,
        normalize=normalize,
        fit_transforms=True,
    )
    transforms = train_dataset.export_transforms()
    val_dataset = AnnualReservoirDataset(
        data_dir=data_cfg.get("train_data_dir", "train"),
        years=[int(y) for y in val_years],
        sequence_length=sequence_length,
        use_log_transform=use_log_transform,
        normalize=normalize,
        fit_transforms=False,
        precomputed_transforms=transforms,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    sample_item = train_dataset[0]
    input_dim = sample_item["features"].shape[-1]
    output_dim = sample_item["targets"].shape[-1]
    multiscale_cfg = cfg.get("multiscale", {}) if cfg else {}
    multiscale_enabled = bool(multiscale_cfg.get("enabled", False))
    multiscale_raw_dim = 0
    multiscale_encoded_dim = 0
    multiscale_channels = 0
    multiscale_gate_logging = False
    ms_tcn: Optional[MultiScaleTCN] = None

    if multiscale_enabled:
        sample_batch_for_ms: Dict[str, torch.Tensor] = {}
        for key, value in sample_item.items():
            if torch.is_tensor(value):
                sample_batch_for_ms[key] = value.unsqueeze(0)
        # Add terminal-guidance context placeholders so channel count matches runtime
        try:
            seq_len_local = int(sample_batch_for_ms["q_min"].size(1))
            R_local = int(sample_batch_for_ms["q_min"].size(2) if sample_batch_for_ms["q_min"].dim() == 3 else sample_batch_for_ms["q_min"].size(1))
        except Exception:
            seq_len_local = int(sequence_length)
            R_local = int(output_dim)
        dt_local = torch.tensor(_TIME_STEP_SECONDS[:seq_len_local], dtype=torch.float32)
        Vmin_local = torch.zeros((1, seq_len_local, R_local), dtype=torch.float32)
        Vmax_local = torch.ones((1, seq_len_local, R_local), dtype=torch.float32)
        sample_batch_for_ms.update({
            "delta_t": dt_local,
            "V_min": Vmin_local,
            "V_max": Vmax_local,
        })
        ms_sample = build_multiscale_features(sample_batch_for_ms, cfg)
        if ms_sample is None:
            multiscale_enabled = False
        else:
            multiscale_channels = ms_sample.shape[-1]
            multiscale_raw_dim = ms_sample.shape[2] * multiscale_channels
            tcn_cfg = multiscale_cfg.get("tcn", {})
            if tcn_cfg.get("enabled", False):
                ms_tcn = MultiScaleTCN(
                    in_dim=multiscale_channels,
                    out_dim=int(tcn_cfg.get("channels", 64)),
                    kernels=tuple(tcn_cfg.get("kernels", [3, 5, 7])),
                    dilations=tuple(tcn_cfg.get("dilations", [1, 2, 3])),
                )
                multiscale_encoded_dim = ms_sample.shape[2] * int(tcn_cfg.get("channels", 64))
            multiscale_gate_logging = bool(multiscale_cfg.get("gating", {}).get("log_branch_weights", False))

    augmented_input_dim = input_dim + multiscale_raw_dim + multiscale_encoded_dim
    extra_feature_dim = augmented_input_dim - input_dim

    model = HierarchicalTransformerPredictor(
        input_dim=augmented_input_dim,
        output_dim=output_dim,
        sequence_length=sequence_length,
        dropout=dropout,
        output_sequence_length=output_sequence_length,
        enable_multiscale=enable_multiscale,
        enable_sequence_decoding=enable_sequence_decoding,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if ms_tcn is not None:
        ms_tcn = ms_tcn.to(device)

    # If disabled, skip hard projection inside the model during training.
    # Inference builds a fresh model from config.yaml and is unaffected.
    if hasattr(model, "enable_hard_projection"):
        try:
            model.enable_hard_projection = bool(model.enable_hard_projection) and train_projection_enabled
        except Exception:
            model.enable_hard_projection = train_projection_enabled

    constraints_cfg = cfg.get("constraints", {}) if cfg else {}
    curves_dir = constraints_cfg.get("curves_dir", str(_SHUXING_DIR / "curves"))
    bounds_dir = constraints_cfg.get("bounds_dir", str(_SHUXING_DIR / "constraints"))
    volume_scale = float(constraints_cfg.get("volume_scale", 1.0))

    reservoirs_order = list(_RESERVOIR_NAMES or _DEFAULT_RESERVOIRS)
    reservoir_names = reservoirs_order
    Vmin_bounds, Vmax_bounds, V0_bounds, VT_bounds, strict_converter = load_and_convert_bounds(
        reservoirs_order,
        constraints_dir=bounds_dir,
        curves_dir=curves_dir,
        volume_scale=volume_scale,
        device=device,
    )

    # ------------------------------------------------------------------
    # Loss, curriculum, and optimizer setup
    # ------------------------------------------------------------------
    loss_cfg = cfg.get("loss", {}) if cfg else {}
    term_cfg = loss_cfg.get("terminal", {})
    ramp_cfg = loss_cfg.get("ramp", {})
    weight_cfg = loss_cfg.get("weights", {})
    power_cfg = loss_cfg.get("power", {})
    barrier_cfg = loss_cfg.get("barrier", {})
    delta_t_train = float(term_cfg.get("delta_t", 864_000.0))
    dt_steps_tensor = torch.tensor(_TIME_STEP_SECONDS, dtype=torch.float32, device=device)

    # Load storage-domain constraints once (Vmin/Vmax and mid-point V0/VT)
    try:
        csv_constraints = _SCRIPT_DIR / "shuxing" / "约束条件.csv"
        names_for_curves = list(_RESERVOIR_NAMES or _DEFAULT_RESERVOIRS)
        curves_dir = str(_SHUXING_DIR / 'curves')
        # Path bounds (Vmin/Vmax) always loaded from CSV
        storage_Vmin, storage_Vmax, _, _ = load_storage_constraints_from_csv(
            csv_constraints,
            names_for_curves,
            periods=36,
            device=device
        )
        cfg = get_config() if CONFIG_AVAILABLE and get_config is not None else None
        initial_storage_cfg = cfg.get("constraints.initial_storage", None) if cfg else None
        target_storage_cfg = cfg.get("constraints.target_storage", None) if cfg else None
        if initial_storage_cfg is None or target_storage_cfg is None:
            raise RuntimeError("config.yaml 缺少 constraints.initial_storage/target_storage")
        V0_arr = torch.tensor(initial_storage_cfg, dtype=torch.float32, device=device).unsqueeze(0)
        VT_arr = torch.tensor(target_storage_cfg, dtype=torch.float32, device=device).unsqueeze(0)
        storage_V0, storage_VT = V0_arr, VT_arr
    except Exception as exc:
        raise RuntimeError(f"加载库容约束失败: {exc}")

    storage_Vmin_seq = storage_Vmin[:sequence_length, :].clone()
    storage_Vmax_seq = storage_Vmax[:sequence_length, :].clone()
    dt_vector_full = torch.tensor(
        _TIME_STEP_SECONDS[:sequence_length], dtype=torch.float32, device=device
    )

    proj_cfg = cfg.get("projection", {}) if cfg else {}
    use_level_projection = bool(proj_cfg.get("enabled", False))
    reach_cfg = proj_cfg.get("reachability", {}) if isinstance(proj_cfg, dict) else {}
    level_limits: Optional[Dict[str, torch.Tensor]] = None
    curves_for_proj: Optional[TorchCurves] = None
    V_T_lo_tensor: Optional[torch.Tensor] = None
    V_T_hi_tensor: Optional[torch.Tensor] = None
    if use_level_projection:
        curves_for_proj = TorchCurves(
            curves_dir=str(_SHUXING_DIR / "curves"),
            tailwater_dir=str(_SHUXING_DIR),
            reservoir_names=reservoir_names,
        ).to(device)
        curves_for_proj.eval()

        ramp_cfg = proj_cfg.get("ramp", {}) if isinstance(proj_cfg, dict) else {}
        ramp_source = str(ramp_cfg.get("source", "none")).lower()
        V_min_base = storage_Vmin_seq
        V_max_base = storage_Vmax_seq
        dyn_h_min = None
        dyn_h_max = None
        dyn_dh_up = None
        dyn_dh_dn = None
        try:
            inflow_year = None
            try:
                if train_years:
                    inflow_year = int(train_years[0])
            except Exception:
                inflow_year = None
            _, _, dyn_constraints = load_dynamic_constraints_and_inflows(
                reservoir_names,
                num_periods=sequence_length,
                inflow_year=inflow_year,
            )
            hmin_list: List[np.ndarray] = []
            hmax_list: List[np.ndarray] = []
            hup_list: List[np.ndarray] = []
            hdn_list: List[np.ndarray] = []
            for name in reservoir_names:
                df_dyn = dyn_constraints.get(name)
                if df_dyn is None:
                    continue
                hmin_list.append(df_dyn["level_min"].to_numpy(dtype=np.float32)[:sequence_length])
                hmax_list.append(df_dyn["level_max"].to_numpy(dtype=np.float32)[:sequence_length])
                hup_list.append(df_dyn["level_up_max"].to_numpy(dtype=np.float32)[:sequence_length])
                hdn_list.append(df_dyn["level_down_max"].to_numpy(dtype=np.float32)[:sequence_length])
            if hmin_list and hmax_list:
                dyn_h_min = torch.tensor(np.stack(hmin_list, axis=1), dtype=torch.float32, device=device)
                dyn_h_max = torch.tensor(np.stack(hmax_list, axis=1), dtype=torch.float32, device=device)
                if hup_list and hdn_list:
                    dyn_dh_up = torch.tensor(np.stack(hup_list, axis=1), dtype=torch.float32, device=device)
                    dyn_dh_dn = torch.tensor(np.stack(hdn_list, axis=1), dtype=torch.float32, device=device)
                    if ramp_source == "none":
                        ramp_source = "config"
                        try:
                            print("[Train] 已自动激活水位变幅约束, source set to 'config'")
                        except Exception:
                            pass
        except Exception as exc:
            print(f"[Train] 警告: 动态约束加载失败 ({exc})，将忽略水位变幅约束。")
            dyn_h_min = None
            dyn_h_max = None
            dyn_dh_up = None
            dyn_dh_dn = None

        H_min_tensor, H_max_tensor, dH_up_tensor, dH_dn_tensor = make_level_limits(
            curves=curves_for_proj,
            V_min=V_min_base,
            V_max=V_max_base,
            H_min=dyn_h_min,
            H_max=dyn_h_max,
            ramp_source=ramp_source,
            ramp_up=dyn_dh_up,
            ramp_dn=dyn_dh_dn,
        )
        level_limits = {
            "H_min": H_min_tensor.to(device),
            "H_max": H_max_tensor.to(device),
            "dH_up": dH_up_tensor.to(device),
            "dH_dn": dH_dn_tensor.to(device),
        }

    flow_loss = WeightedMSELoss(**loss_cfg.get("flow", {}))
    terminal_loss = None
    if term_cfg.get("enabled", True):
        terminal_loss = TerminalWindowLoss(
            window_k=term_cfg.get("window_k", 12),
            use_level=term_cfg.get("use_level", False),
            eps=term_cfg.get("eps", 1e-8),
        )
    ramp_loss = None
    if ramp_cfg.get("enabled", False):
        ramp_loss = RampLoss(weight=ramp_cfg.get("weight", 0.0))
    barrier_loss = None
    if barrier_cfg.get("enabled", False):
        barrier_loss = InteriorBarrierLoss(eps=barrier_cfg.get("eps", 1e-6))

    power_surrogate = None
    term_tail_profile = term_cfg.get("tail_profile", "linear")
    term_window_override = term_cfg.get("per_reservoir_window_k", [])
    late_margin_window_override = term_cfg.get("late_margin_window_k", [])
    term_to_go_reservoir_weights = weight_cfg.get("term_to_go_reservoir")
    late_margin_reservoir_weights = weight_cfg.get("late_margin_reservoir")
    late_margin_values_raw = power_cfg.get("late_margin_m3_per_reservoir")
    late_margin_default_raw = power_cfg.get("late_margin_m3")
    if late_margin_default_raw is None:
        if isinstance(late_margin_values_raw, (list, tuple)) and late_margin_values_raw:
            late_margin_default_raw = late_margin_values_raw[0]
        else:
            late_margin_default_raw = 0.0
    late_margin_default = float(late_margin_default_raw) / 1e8
    late_margin_values_scaled = None
    if isinstance(late_margin_values_raw, (list, tuple)):
        late_margin_values_scaled = [float(x) / 1e8 for x in late_margin_values_raw]

    path_cfg = loss_cfg.get("path", {})
    w_path = float(weight_cfg.get("path", 0.0))
    water_level_cfg = loss_cfg.get("water_level", {}) if hasattr(loss_cfg, "get") else {}
    reserve_cfg = loss_cfg.get("reserve", {}) if hasattr(loss_cfg, "get") else {}

    reservoir_names = list(_RESERVOIR_NAMES or _DEFAULT_RESERVOIRS)
    curves_module = None
    need_curves = (
        water_level_cfg.get("w_path", 0.0) > 0.0
        or water_level_cfg.get("w_ramp", 0.0) > 0.0
        or (reserve_cfg.get("enabled", False) and reserve_cfg.get("weight", 0.0) > 0.0)
        or (weight_cfg.get("power", 0.0) > 0.0 and power_cfg.get("enabled", enable_power_optimization))
    )
    if need_curves:
        try:
            curves_module = TorchCurves(
                curves_dir=str(_SHUXING_DIR / "curves"),
                tailwater_dir=str(_SHUXING_DIR),
                reservoir_names=reservoir_names,
            ).to(device)
        except Exception as exc:
            print(f"[warn] failed to load curves: {exc}")
            curves_module = None

    # Parse per-reservoir barrier weights from config mapping.
    def _per_reservoir_list(mapping: Any, names: Sequence[str], default_value: float = 1.0) -> Optional[List[float]]:
        if not isinstance(mapping, dict) or not names:
            return None
        default_scalar = mapping.get("default", default_value)
        try:
            base = float(default_scalar)
        except (TypeError, ValueError):
            base = float(default_value)
        values: List[float] = []
        for name in names:
            raw = mapping.get(name, base)
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                values.append(base)
        return values

    def _per_reservoir_level_targets(h_cfg: Any, names: Sequence[str], default_value: float) -> Optional[List[float]]:
        if not isinstance(h_cfg, dict) or not names:
            return None
        parsed: List[float] = []
        for name in names:
            candidate = h_cfg.get(name, {})
            value = default_value
            if isinstance(candidate, dict):
                value = candidate.get("value", value)
            elif candidate is not None:
                value = candidate
            try:
                parsed.append(float(value))
            except (TypeError, ValueError):
                parsed.append(float(default_value))
        return parsed

    level_path_weights = _per_reservoir_list(
        water_level_cfg.get("per_reservoir_mult", {}), reservoir_names, default_value=1.0
    )
    level_ramp_weights = level_path_weights

    reserve_enabled = bool(reserve_cfg.get("enabled", False))
    w_reserve = float(reserve_cfg.get("weight", 0.0))
    reserve_k_tail = int(reserve_cfg.get("k_tail", 0))
    reserve_margin = float(reserve_cfg.get("margin", 0.0))
    reserve_weights = _per_reservoir_list(
        reserve_cfg.get("per_reservoir_mult", {}), reservoir_names, default_value=1.0
    )

    term_mode = str(term_cfg.get("mode", "volume_window"))
    h_target_cfg = term_cfg.get("h_target", {}) if hasattr(term_cfg, "get") else {}
    cfg_constraints = cfg.get("constraints", {}) if hasattr(cfg, "get") else {}
    target_levels_cfg = cfg_constraints.get("target_levels", []) if hasattr(cfg_constraints, "get") else []
    constraint_target_levels: List[float] = []
    for idx, _ in enumerate(reservoir_names):
        try:
            constraint_target_levels.append(float(target_levels_cfg[idx]))
        except Exception:
            constraint_target_levels.append(0.0)
    default_h_target = constraint_target_levels[0] if constraint_target_levels else 0.0
    if isinstance(h_target_cfg, dict):
        default_entry = h_target_cfg.get("default", {})
        if isinstance(default_entry, dict):
            default_h_target = float(default_entry.get("value", default_h_target))
        elif default_entry not in (None, {}):
            try:
                default_h_target = float(default_entry)
            except (TypeError, ValueError):
                default_h_target = constraint_target_levels[0] if constraint_target_levels else 0.0
    terminal_level_targets = []
    for idx, name in enumerate(reservoir_names):
        fallback_level = constraint_target_levels[idx] if idx < len(constraint_target_levels) else default_h_target
        entry = h_target_cfg.get(name, None) if isinstance(h_target_cfg, dict) else None
        if isinstance(entry, dict):
            terminal_level_targets.append(float(entry.get("value", fallback_level)))
        elif entry not in (None, {}):
            try:
                terminal_level_targets.append(float(entry))
            except (TypeError, ValueError):
                terminal_level_targets.append(float(fallback_level))
        else:
            terminal_level_targets.append(float(fallback_level))
    terminal_level_weights = _per_reservoir_list(
        term_cfg.get("per_reservoir_mult", {}), reservoir_names, default_value=1.0
    )
    need_reservoir_weights = _per_reservoir_list(
        term_cfg.get("need_per_reservoir_mult", {}), reservoir_names, default_value=1.0
    )

    proj_guidance_cfg = loss_cfg.get("proj_guidance", {}) if hasattr(loss_cfg, "get") else {}
    proj_guidance_enabled = bool(proj_guidance_cfg.get("enabled", False))
    proj_guidance_weight = float(proj_guidance_cfg.get("weight", 0.0))
    proj_guidance_tail = max(0, int(proj_guidance_cfg.get("tail_periods", 0)))
    guidance_reservoirs = proj_guidance_cfg.get("reservoirs", []) or []
    proj_guidance_indices: List[int] = []
    for name in guidance_reservoirs:
        if name in reservoir_names:
            proj_guidance_indices.append(reservoir_names.index(name))

    cascade_consistency_cfg = training_cfg.get("cascade_consistency", {}) if isinstance(training_cfg, dict) else {}
    cascade_consistency_enabled = bool(cascade_consistency_cfg.get("enabled", multiscale_enabled))
    cascade_consistency_weight = float(cascade_consistency_cfg.get("weight", 0.0))
    if not multiscale_enabled:
        cascade_consistency_enabled = False
        cascade_consistency_weight = 0.0

    criterion = CompositeSchedulingLoss(
        flow_loss=flow_loss,
        power_surrogate=power_surrogate,
        w_flow=weight_cfg.get("flow", 1.0),
        w_term=weight_cfg.get("term", 1.0),
        w_ramp=weight_cfg.get("ramp", 0.0),
        w_power=weight_cfg.get("power", power_weight if enable_power_optimization else 0.0),
        terminal_loss=terminal_loss,
        ramp_loss=ramp_loss,
        barrier_loss=barrier_loss,
        w_barrier=weight_cfg.get("barrier", barrier_cfg.get("weight", 0.0)),
        w_need=weight_cfg.get("need", 0.0),
        w_path=w_path,
        path_mode=path_cfg.get("mode", "hinge"),
        w_term_to_go=weight_cfg.get("term_to_go", 0.0),
        w_late_margin=weight_cfg.get("late_margin", 0.0),
        late_margin_value=late_margin_default,
        term_to_go_reservoir_weights=term_to_go_reservoir_weights,
        term_to_go_window_override=term_window_override,
        late_margin_reservoir_weights=late_margin_reservoir_weights,
        late_margin_values=late_margin_values_scaled,
        late_margin_window_override=late_margin_window_override,
        tail_weight_profile=term_tail_profile,
        water_level_curves=curves_module,
        w_level_path=float(water_level_cfg.get("w_path", 0.0)),
        w_level_ramp=float(water_level_cfg.get("w_ramp", 0.0)),
        level_path_reservoir_weights=level_path_weights,
        level_ramp_reservoir_weights=level_ramp_weights,
        terminal_mode=term_mode,
        terminal_level_target=terminal_level_targets,
        terminal_level_tolerance=float(term_cfg.get("h_tolerance", 0.0)),
        terminal_level_reservoir_weights=terminal_level_weights,
        reserve_enabled=reserve_enabled,
        w_reserve=w_reserve,
        reserve_k_tail=reserve_k_tail,
        reserve_margin=reserve_margin,
        reserve_reservoir_weights=reserve_weights,
        need_reservoir_weights=need_reservoir_weights,
    )
    # Optionally keep path/barrier loss even with hard projection
    allow_path_with_proj = bool(path_cfg.get("allow_with_projection", False))
    if getattr(model, "enable_hard_projection", False):
        if not allow_path_with_proj:
            criterion.w_path = 0.0
        # Barrier often redundant with projection; keep configurable similarly
        barrier_cfg_local = loss_cfg.get("barrier", {}) if hasattr(loss_cfg, "get") else {}
        if not bool(barrier_cfg_local.get("allow_with_projection", False)):
            criterion.w_barrier = 0.0

    power_energy_module: Optional[PowerEnergySurrogate] = None
    if criterion.w_power > 0.0 and power_cfg.get("enabled", enable_power_optimization):
        if curves_module is None:
            print("[warn] power surrogate disabled because curves are unavailable")
            criterion.w_power = 0.0
        else:
            try:
                power_eta = float(power_cfg.get("eta", 0.88))
                power_energy_module = PowerEnergySurrogate(curves_module, eta=power_eta).to(device)
                power_energy_module.eval()
                per_reservoir_scale = power_cfg.get("per_reservoir_scale")
                if per_reservoir_scale is not None:
                    scale_tensor = torch.tensor(per_reservoir_scale, dtype=torch.float32, device=device)
                    R_len = len(reservoir_names)
                    if scale_tensor.numel() < R_len:
                        pad_value = (
                            scale_tensor[-1]
                            if scale_tensor.numel() > 0
                            else torch.tensor(1.0, device=device, dtype=torch.float32)
                        )
                        pad_count = R_len - scale_tensor.numel()
                        pad_tensor = torch.full(
                            (pad_count,),
                            float(pad_value.item()),
                            dtype=scale_tensor.dtype,
                            device=device,
                        )
                        scale_tensor = torch.cat([scale_tensor, pad_tensor], dim=0)
                    elif scale_tensor.numel() > R_len:
                        scale_tensor = scale_tensor[:R_len]
                    power_energy_module.set_per_reservoir_scale(scale_tensor)
            except Exception as exc:
                print(f"[warn] power surrogate disabled due to error: {exc}")
                power_energy_module = None
                criterion.w_power = 0.0

    curriculum_cfg = cfg.get("curriculum", {}) if cfg else {}
    decoder_cfg = cfg.get("decoder", {}) if cfg else {}
    metrics_cfg = cfg.get("metrics", {}) if cfg else {}

    # Strict level-volume converter loaded from shuxing/curves; missing files raise errors.
    strict_conv = strict_converter

    def to_level_fn(storage: torch.Tensor) -> torch.Tensor:
        outs = []
        for r in range(storage.size(-1)):
            outs.append(strict_conv.s2h(storage[..., r], r))
        return torch.stack(outs, dim=-1)

    def to_volume_fn(levels: torch.Tensor) -> torch.Tensor:
        outs = []
        for r in range(levels.size(-1)):
            outs.append(strict_conv.h2s(levels[..., r], r))
        return torch.stack(outs, dim=-1)

    print(
        "[Loss] init weights -> "
        f"w_flow={criterion.w_flow:.3f} w_term={criterion.w_term:.3f} "
        f"w_ramp={criterion.w_ramp:.3f} w_power={criterion.w_power:.3f} "
        f"w_barrier={criterion.w_barrier:.3f}"
    )
    if log_mode == "standard":
        print("[Loss] logging mode=standard, normalized loss = |loss| / |epoch1_loss|")
    if criterion.terminal_loss is not None:
        print(
            f"[Loss] terminal K={criterion.terminal_loss.window_k}"
        )

    trainable_params: List[torch.nn.Parameter] = list(model.parameters())
    if ms_tcn is not None:
        trainable_params.extend(list(ms_tcn.parameters()))
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)
    scheduler_cfg: Dict[str, Any] = {}
    if isinstance(training_cfg, dict):
        scheduler_cfg = training_cfg.get("scheduler", {}) or {}
    scheduler_patience = int(best_params.get(
        "scheduler_patience",
        scheduler_cfg.get("patience", max(2, patience // 2)),
    ))
    scheduler_kwargs = {
        "mode": scheduler_cfg.get("mode", "min"),
        "factor": float(scheduler_cfg.get("factor", 0.5)),
        "patience": max(1, scheduler_patience),
        "threshold": float(scheduler_cfg.get("threshold", 1e-4)),
        "threshold_mode": scheduler_cfg.get("threshold_mode", "rel"),
        "cooldown": int(scheduler_cfg.get("cooldown", 0)),
        "min_lr": float(scheduler_cfg.get("min_lr", 0.0)),
        "eps": float(scheduler_cfg.get("eps", 1e-8)),
    }
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        **scheduler_kwargs,
    )

    trainer_smoke = False
    if hasattr(cfg, "get"):
        try:
            trainer_smoke = bool(cfg.get("trainer.smoke", False))
        except Exception:
            trainer_smoke = False
    elif isinstance(cfg, dict):
        trainer_smoke = bool(cfg.get("trainer", {}).get("smoke", False))

    def _prepare_features_tensor(
        batch: Dict[str, torch.Tensor],
        qmin_tensor: torch.Tensor,
        qmax_tensor: torch.Tensor,
        *,
        is_training: bool,
        step: Optional[int] = None,
        q_in_override: Optional[torch.Tensor] = None,
        q_out_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        nonlocal ms_tcn
        feats = batch["features"].to(device)
        if not multiscale_enabled:
            return feats

        # Build multiscale + terminal-guidance features
        steps_local = qmin_tensor.size(1)
        dt_local = dt_vector_full[:steps_local]
        Vmin_local = storage_Vmin_seq[:steps_local, :]
        Vmax_local = storage_Vmax_seq[:steps_local, :]
        q_in_tensor = q_in_override if q_in_override is not None else batch["q_in"].to(device)
        ms_inputs = {
            "q_in": q_in_tensor,
            "q_min": qmin_tensor,
            "q_max": qmax_tensor,
            "delta_t": dt_local,
            "V0": batch["V0"].to(device),
            "V_target": batch["V_target"].to(device),
            "V_min": Vmin_local,
            "V_max": Vmax_local,
        }
        if q_out_override is not None:
            ms_inputs["q_out_pred"] = q_out_override

        ms_stats = build_multiscale_features(ms_inputs, cfg)
        if ms_stats is None:
            return feats

        ms_stats = ms_stats.to(device)
        parts: List[torch.Tensor] = [feats, ms_stats.reshape(ms_stats.size(0), ms_stats.size(1), -1)]

        if ms_tcn is not None:
            # Adapt TCN if input channel count changed due to added guidance features
            try:
                current_in = getattr(ms_tcn.branches[0][0], "in_channels", None)
                new_in = int(ms_stats.size(-1))
                if current_in is not None and current_in != new_in:
                    ms_tcn = MultiScaleTCN(
                        in_dim=new_in,
                        out_dim=tcn_channels,
                        kernels=tcn_kernels,
                        dilations=tcn_dilations,
                    ).to(device)
                    ms_tcn.eval() if not is_training else None
            except Exception:
                pass
            ms_encoded, gate_weights = ms_tcn(ms_stats)
            parts.append(ms_encoded.reshape(ms_encoded.size(0), ms_encoded.size(1), -1))
            if multiscale_gate_logging and is_training and step is not None and step % 200 == 0:
                try:
                    print(f"[multiscale] step={step} gate={gate_weights.tolist()}")
                except Exception:
                    pass

        return torch.cat(parts, dim=-1)

    def _cascade_consistency_forward(
        batch: Dict[str, torch.Tensor],
        qmin_tensor: torch.Tensor,
        qmax_tensor: torch.Tensor,
        q_in_consistent: torch.Tensor,
        q_out_consistent: torch.Tensor,
        V0_tensor: torch.Tensor,
        V_target_tensor: torch.Tensor,
        dt_tensor: torch.Tensor,
        Vmin_tensor: torch.Tensor,
        Vmax_tensor: torch.Tensor,
        *,
        is_training: bool,
        step: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if not (cascade_consistency_enabled and cascade_consistency_weight > 0.0 and multiscale_enabled):
            return None
        q_in_aux = q_in_consistent.detach()
        q_out_aux = q_out_consistent.detach()
        features_consistent = _prepare_features_tensor(
            batch,
            qmin_tensor,
            qmax_tensor,
            is_training=is_training,
            step=step,
            q_in_override=q_in_aux,
            q_out_override=q_out_aux,
        )
        out_consistent = model(
            features_consistent,
            q_min=qmin_tensor,
            q_max=qmax_tensor,
            q_in=q_in_aux,
            V0=V0_tensor,
            V_target=V_target_tensor,
            delta_t=dt_tensor,
            V_min=Vmin_tensor,
            V_max=Vmax_tensor,
        )
        return out_consistent if not isinstance(out_consistent, tuple) else out_consistent[0]

    if trainer_smoke:
        smoke_loader = build_dataloaders(split="train", batch_size=2, num_workers=0)
        smoke_batch = next(iter(smoke_loader))
        model.eval()
        if ms_tcn is not None:
            ms_tcn.eval()
        with torch.no_grad():
            qmin_smoke = smoke_batch["q_min"].to(device)
            qmax_smoke = smoke_batch["q_max"].to(device)
            q_in_smoke = smoke_batch["q_in"].to(device)
            V0_smoke = smoke_batch["V0"].to(device)
            V_target_smoke = smoke_batch["V_target"].to(device)
            features_smoke = _prepare_features_tensor(
                smoke_batch,
                qmin_smoke,
                qmax_smoke,
                is_training=False,
            )
            steps_smoke = qmin_smoke.size(1)
            dt_vec_smoke = dt_vector_full[:steps_smoke]
            Vmin_smoke = storage_Vmin_seq[:steps_smoke, :]
            Vmax_smoke = storage_Vmax_seq[:steps_smoke, :]

            q_phys_smoke = model(
                features_smoke,
                q_min=qmin_smoke,
                q_max=qmax_smoke,
                q_in=q_in_smoke,
                V0=V0_smoke,
                V_target=V_target_smoke,
                delta_t=dt_vec_smoke,
                V_min=Vmin_smoke,
                V_max=Vmax_smoke,
            )
            violation_smoke = ((q_phys_smoke < qmin_smoke) | (q_phys_smoke > qmax_smoke)).float().mean().item()

            dt_vol_smoke = dt_vec_smoke.to(q_phys_smoke.dtype).view(1, -1, 1) / 1e8
            V_path_smoke = V0_smoke.unsqueeze(1) + torch.cumsum((q_in_smoke - q_phys_smoke) * dt_vol_smoke, dim=1)
            V_T_smoke = V_path_smoke[:, -1, :]
            terminal_smoke = (V_target_smoke - V_T_smoke).abs().mean().item()

        print(f"[smoke] violation={violation_smoke:.6g}, terminal_error={terminal_smoke:.6g}")
        if violation_smoke > 0:
            raise RuntimeError("Mapped outputs must be within [q_min,q_max].")
        model.train()
        if ms_tcn is not None:
            ms_tcn.train()

    def _schedule_progress(epoch_idx: int, schedule_cfg: Dict[str, Any]) -> float:
        start_epoch = int(schedule_cfg.get("start_epoch", 0))
        end_epoch = int(schedule_cfg.get("end_epoch", max(1, max_epochs - 1)))
        if end_epoch <= start_epoch:
            return 1.0 if epoch_idx >= start_epoch else 0.0
        if epoch_idx <= start_epoch:
            return 0.0
        if epoch_idx >= end_epoch:
            return 1.0
        return (epoch_idx - start_epoch) / max(1, end_epoch - start_epoch)

    def _apply_curriculum(epoch_idx: int) -> None:
        if max_epochs <= 1:
            progress = 1.0
        else:
            progress = epoch_idx / max(1, max_epochs - 1)

        term_sched = curriculum_cfg.get("term", {})
        if term_sched:
            term_progress = _schedule_progress(epoch_idx, term_sched)
            criterion.w_term = _lerp(term_sched.get("w_start", criterion.w_term), term_sched.get("w_end", criterion.w_term), term_progress)

        flow_sched = curriculum_cfg.get("flow", {})
        if flow_sched:
            flow_progress = _schedule_progress(epoch_idx, flow_sched)
            criterion.w_flow = _lerp(flow_sched.get("w_start", criterion.w_flow), flow_sched.get("w_end", criterion.w_flow), flow_progress)

        power_sched = curriculum_cfg.get("power", {})
        if power_sched:
            power_progress = _schedule_progress(epoch_idx, power_sched)
            criterion.w_power = _lerp(power_sched.get("w_start", criterion.w_power), power_sched.get("w_end", criterion.w_power), power_progress)

        if criterion.terminal_loss is not None:
            k_sched = curriculum_cfg.get("k", {})
            if k_sched:
                criterion.terminal_loss.window_k = int(round(_lerp(k_sched.get("start", criterion.terminal_loss.window_k), k_sched.get("end", criterion.terminal_loss.window_k), progress)))

    def _terminal_metrics_raw(
        q_out: torch.Tensor,
        q_in: torch.Tensor,
        V0_tensor: torch.Tensor,
        dt_steps: torch.Tensor,
        V_target_tensor: torch.Tensor,
    ) -> Tuple[float, float]:
        if q_out is None:
            return float("nan"), float("nan")
        dt = dt_steps.to(dtype=q_out.dtype, device=q_out.device).view(1, -1, 1) / 1e8
        V_path = V0_tensor.unsqueeze(1) + torch.cumsum((q_in - q_out) * dt, dim=1)
        V_T = V_path[:, -1, :]
        V_target_expand = V_target_tensor
        if V_target_expand.dim() == 1:
            V_target_expand = V_target_expand.unsqueeze(0)
        if V_target_expand.size(0) == 1 and V_T.size(0) > 1:
            V_target_expand = V_target_expand.expand(V_T.size(0), -1)
        gap = (V_target_expand - V_T).abs()
        return float(gap.mean().item()), float(gap.max().item())

    best_state = copy.deepcopy(model.state_dict())
    best_ms_state = copy.deepcopy(ms_tcn.state_dict()) if ms_tcn is not None else None
    best_val_loss = float("inf")
    best_score = -float("inf")
    epochs_no_improve = 0

    term_coef = float(metrics_cfg.get("term_coef", 1.0))
    violation_coef = float(metrics_cfg.get("violation_coef", 10.0))
    power_coef = float(metrics_cfg.get("power_coef", 0.1))
    ramp_coef = float(metrics_cfg.get("ramp_coef", 0.0))
    selection_mode = str(metrics_cfg.get("selection_mode", "balanced"))
    feasible_violation_tol = float(metrics_cfg.get("feasible_violation_tolerance", 0.0))
    feasible_infeasible_tol = float(metrics_cfg.get("feasible_infeasible_tolerance", 0.0))
    feasible_term_tol = float(metrics_cfg.get("feasible_term_error_tolerance", 1.0e9))
    infeasible_penalty = float(metrics_cfg.get("infeasible_penalty", 100.0))
    infeasible_warn_ratio = float(metrics_cfg.get("infeasible_warn_ratio", 0.0))

    val_preds: List[np.ndarray] = []
    val_targets: List[np.ndarray] = []
    global_step = 0
    train_loss_ref: Optional[float] = None
    val_loss_ref: Optional[float] = None
    loss_history_rows: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(1, max_epochs + 1):
        _apply_curriculum(epoch - 1)
        model.train()
        if ms_tcn is not None:
            ms_tcn.train()
        train_loss_sum = 0.0
        train_sq_err_sum = 0.0
        train_abs_target_sum = 0.0
        train_elem_count = 0
        raw_need_epoch_mean = float("nan")
        raw_need_epoch_max = float("nan")

        for batch_idx, batch in enumerate(train_loader):
            qmin = batch["q_min"].to(device)
            qmax = batch["q_max"].to(device)
            features = _prepare_features_tensor(batch, qmin, qmax, is_training=True, step=global_step)
            targets_phys = batch["original_outflow"].to(device)
            q_in_obs = batch["q_in"].to(device)
            head_inflow = batch["head_inflow"].to(device)
            interval_inflow = batch["interval_inflow"].to(device)
            V0 = batch["V0"].to(device)
            V_target = batch["V_target"].to(device)
            steps = features.size(1)
            dt_vec = dt_vector_full[:steps]
            Vmin_use = storage_Vmin_seq[:steps, :]
            Vmax_use = storage_Vmax_seq[:steps, :]
            terminal_policy = _resolve_terminal_policy_targets(
                q_min=qmin,
                q_max=qmax,
                head_inflow=head_inflow,
                interval_inflow=interval_inflow,
                V0=V0,
                delta_t=dt_vec,
                V_min=Vmin_use,
                V_max=Vmax_use,
                V_target=V_target,
                reservoir_names=reservoir_names,
            )
            terminal_reachable_mask = terminal_policy["reachable_mask"]
            terminal_effective_target = terminal_policy["effective_target"]

            optimizer.zero_grad()
            out = model(
                features,
                q_min=qmin,
                q_max=qmax,
                q_in=q_in_obs,
                V0=V0,
                V_target=V_target,
                delta_t=dt_vec,
                V_min=Vmin_use,
                V_max=Vmax_use,
                terminal_reachable_mask=terminal_reachable_mask,
                terminal_best_effort_target=terminal_effective_target,
                return_logits=True,
            )
            if isinstance(out, tuple) and len(out) == 3:
                q_phys, raw_logits, q_base = out
            else:
                # fallback for safety
                q_phys = out if not isinstance(out, tuple) else out[0]
                raw_logits = None
                q_base = q_phys
            level_limits_slice = None
            if train_projection_enabled and use_level_projection and level_limits is not None and curves_for_proj is not None:
                level_limits_slice = {k: v[:steps, :] for k, v in level_limits.items()}

            def _train_post_step(q_cur: torch.Tensor, q_in_cur: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
                if level_limits_slice is None or curves_for_proj is None:
                    return q_cur, {}
                q_proj, _, _ = _apply_level_projection(
                    q_raw=q_cur,
                    q_in=q_in_cur,
                    V0=V0,
                    dt_seconds=dt_vec,
                    q_min=qmin,
                    q_max=qmax,
                    level_limits=level_limits_slice,
                    curves=curves_for_proj,
                    V_T_lo=V_T_lo_tensor,
                    V_T_hi=V_T_hi_tensor,
                    reach_cfg=reach_cfg,
                )
                return q_proj, {}

            sc_result = _run_self_consistent_refinement(
                q_phys=q_phys,
                raw_logits=raw_logits,
                model=model,
                head_inflow=head_inflow,
                interval_inflow=interval_inflow,
                q_min=qmin,
                q_max=qmax,
                V0=V0,
                V_target=terminal_effective_target,
                delta_t=dt_vec,
                V_min=Vmin_use,
                V_max=Vmax_use,
                sc_iters=sc_iters_train,
                post_step=_train_post_step,
            )
            q_phys = sc_result["q_phys"]
            joint_vmin, joint_vmax = _joint_storage_bounds(
                Vmin_use, Vmax_use, level_limits_slice, curves_for_proj
            )
            flow_ramps = _CONSTRAINT_CFG.get("ramp_constraints", {})
            q_phys, joint_storage = project_joint_schedule(
                q_proposed=q_phys,
                head_inflow=head_inflow,
                interval_inflow=interval_inflow,
                V0=V0,
                V_target=terminal_effective_target,
                q_min=qmin,
                q_max=qmax,
                V_min=joint_vmin,
                V_max=joint_vmax,
                delta_t=dt_vec,
                flow_ramp_up=flow_ramps.get("ramp_up"),
                flow_ramp_down=flow_ramps.get("ramp_down"),
            )
            q_in_pred = _compute_cascade_inflows(q_phys, head_inflow, interval_inflow)
            model.last_projection = {"V": joint_storage, "meta": None}
            # Use storage-based terminal targets for tail losses.
            V0_use = storage_V0.expand(V0.shape[0], -1)
            V_target_use = V_target
            # Provide per-period delta_t to the loss in seconds.
            dt_vec_loss = dt_vec.to(dtype=q_phys.dtype, device=q_phys.device)
            Vmin_loss = Vmin_use.to(dtype=q_phys.dtype, device=q_phys.device)
            Vmax_loss = Vmax_use.to(dtype=q_phys.dtype, device=q_phys.device)
            power_value = None
            if power_energy_module is not None and criterion.w_power > 0.0:
                try:
                    power_scores = power_energy_module(
                        q_out=q_phys,
                        q_in=q_in_pred,
                        V0=V0_use,
                        dt_steps=dt_vec_loss,
                    )
                    power_value = power_scores.mean()
                except Exception as exc:
                    if global_step == 0:
                        print(f"[warn] power surrogate evaluation failed: {exc}")
                    power_value = None
            out_loss = criterion(
                q_phys,
                targets_phys,
                physical_predictions=q_phys,
                q_in=q_in_pred,
                V0=V0_use,
                V_target=V_target_use,
                q_min=qmin,
                q_max=qmax,
                window_k=criterion.terminal_loss.window_k if criterion.terminal_loss is not None else None,
                to_level=to_level_fn,
                delta_t_steps=dt_vec_loss,
                V_min=Vmin_loss,
                V_max=Vmax_loss,
                terminal_effective_target=terminal_effective_target,
                return_term_per_sample=True,
            )
            if isinstance(out_loss, tuple) and len(out_loss) == 2:
                base_loss, term_per_sample = out_loss
            else:
                base_loss = out_loss
                term_per_sample = None
            loss = base_loss
            # Apply focal-style reweighting only to terminal component if available
            if term_per_sample is not None and getattr(criterion, "w_term", 0.0) > 0.0:
                term_mean = term_per_sample.mean()
                gamma = 1.0
                with torch.no_grad():
                    weights = (term_per_sample ** gamma)
                    weights = weights / (weights.mean() + 1e-8)
                loss_term_weighted = (weights * term_per_sample).mean()
                lambda_term = float(getattr(criterion, "w_term", 0.0))
                loss = loss - lambda_term * term_mean + lambda_term * loss_term_weighted
            if power_value is not None:
                loss = loss - criterion.w_power * power_value
            consistency_loss = q_phys.new_tensor(0.0)
            q_consistent = _cascade_consistency_forward(
                batch,
                qmin,
                qmax,
                q_in_pred,
                q_phys,
                V0,
                V_target,
                dt_vec,
                Vmin_use,
                Vmax_use,
                is_training=True,
                step=global_step,
            )
            if q_consistent is not None:
                consistency_loss = torch.nn.functional.mse_loss(q_consistent, q_phys.detach())
                loss = loss + cascade_consistency_weight * consistency_loss
            # End-to-end differentiable training:
            # The main loss 'criterion' already uses q_phys (projected flow).
            # Since _apply_level_projection is differentiable, we don't need 'projection imitation'.
            # Gradients will flow directly from the constraints (via q_phys) back to the model.
            
            if proj_guidance_enabled:
                # Optional: keep guidance loss but use q_phys directly if needed, 
                # or remove it if we trust the end-to-end gradient.
                # For now, let's disable the imitation/guidance part to let the physics loss drive the learning.
                pass
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            optimizer.step()
            train_loss_sum += loss.item() * features.size(0)
            err_train = q_phys.detach() - targets_phys.detach()
            train_sq_err_sum += float((err_train * err_train).sum().item())
            train_abs_target_sum += float(targets_phys.detach().abs().sum().item())
            train_elem_count += int(targets_phys.numel())
            if batch_idx == 0:
                raw_need_epoch_mean, raw_need_epoch_max = _terminal_metrics_raw(
                    q_base.detach(),
                    q_in_obs.detach(),
                    V0.detach(),
                    dt_vec_loss,
                    V_target.detach(),
                )
            global_step += 1

        train_loss = train_loss_sum / max(1, len(train_loader.dataset))
        train_rmse = math.sqrt(train_sq_err_sum / max(1, train_elem_count))
        train_mean_abs_target = train_abs_target_sum / max(1, train_elem_count)
        train_nrmse = train_rmse / max(1e-8, train_mean_abs_target)

        model.eval()
        if ms_tcn is not None:
            ms_tcn.eval()
        val_loss_sum = 0.0
        val_sq_err_sum = 0.0
        val_abs_target_sum = 0.0
        val_elem_count = 0
        val_violation_sum = 0.0
        val_term_err_sum = 0.0
        val_ramp_sum = 0.0
        val_power_sum = 0.0
        val_barrier_sum = 0.0
        val_consistency_sum = 0.0
        val_count = 0
        val_preds_epoch: List[np.ndarray] = []
        val_targets_epoch: List[np.ndarray] = []
        infeasible_num = 0.0
        total_targets = 0.0
        need_values: List[float] = []
        cap_values: List[float] = []
        delta_t_eval = dt_vector_full
        window_k_eval = (
            int(criterion.terminal_loss.window_k)
            if criterion.terminal_loss is not None
            else int(term_cfg.get("window_k", 0))
        )

        raw_need_val: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                qmin = batch["q_min"].to(device)
                qmax = batch["q_max"].to(device)
                features = _prepare_features_tensor(batch, qmin, qmax, is_training=False)
                targets_phys = batch["original_outflow"].to(device)
                q_in_obs = batch["q_in"].to(device)
                head_inflow = batch["head_inflow"].to(device)
                interval_inflow = batch["interval_inflow"].to(device)
                V0 = batch["V0"].to(device)
                V_target = batch["V_target"].to(device)
                steps = features.size(1)
                dt_vec = dt_vector_full[:steps]
                Vmin_use = storage_Vmin_seq[:steps, :]
                Vmax_use = storage_Vmax_seq[:steps, :]
                terminal_policy = _resolve_terminal_policy_targets(
                    q_min=qmin,
                    q_max=qmax,
                    head_inflow=head_inflow,
                    interval_inflow=interval_inflow,
                    V0=V0,
                    delta_t=dt_vec,
                    V_min=Vmin_use,
                    V_max=Vmax_use,
                    V_target=V_target,
                    reservoir_names=reservoir_names,
                )
                terminal_reachable_mask = terminal_policy["reachable_mask"]
                terminal_effective_target = terminal_policy["effective_target"]

                out = model(
                    features,
                    q_min=qmin,
                    q_max=qmax,
                    q_in=q_in_obs,
                    V0=V0,
                    V_target=V_target,
                    delta_t=dt_vec,
                    V_min=Vmin_use,
                    V_max=Vmax_use,
                    terminal_reachable_mask=terminal_reachable_mask,
                    terminal_best_effort_target=terminal_effective_target,
                    return_logits=True,
                )
                if isinstance(out, tuple) and len(out) == 3:
                    q_phys, raw_logits, q_base_val = out
                else:
                    q_phys = out if not isinstance(out, tuple) else out[0]
                    raw_logits = None
                    q_base_val = q_phys
                level_limits_slice = None
                if train_projection_enabled and use_level_projection and level_limits is not None and curves_for_proj is not None:
                    level_limits_slice = {k: v[:steps, :] for k, v in level_limits.items()}

                def _val_post_step(q_cur: torch.Tensor, q_in_cur: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
                    if level_limits_slice is None or curves_for_proj is None:
                        return q_cur, {}
                    q_proj, _, _ = _apply_level_projection(
                        q_raw=q_cur,
                        q_in=q_in_cur,
                        V0=V0,
                        dt_seconds=dt_vec,
                        q_min=qmin,
                        q_max=qmax,
                        level_limits=level_limits_slice,
                        curves=curves_for_proj,
                        V_T_lo=V_T_lo_tensor,
                        V_T_hi=V_T_hi_tensor,
                        reach_cfg=reach_cfg,
                    )
                    return q_proj, {}

                sc_result = _run_self_consistent_refinement(
                    q_phys=q_phys,
                    raw_logits=raw_logits,
                    model=model,
                    head_inflow=head_inflow,
                    interval_inflow=interval_inflow,
                    q_min=qmin,
                    q_max=qmax,
                    V0=V0,
                    V_target=terminal_effective_target,
                    delta_t=dt_vec,
                    V_min=Vmin_use,
                    V_max=Vmax_use,
                    sc_iters=sc_iters_val,
                    post_step=_val_post_step,
                )
                q_phys = sc_result["q_phys"]
                joint_vmin, joint_vmax = _joint_storage_bounds(
                    Vmin_use, Vmax_use, level_limits_slice, curves_for_proj
                )
                flow_ramps = _CONSTRAINT_CFG.get("ramp_constraints", {})
                q_phys, joint_storage = project_joint_schedule(
                    q_proposed=q_phys,
                    head_inflow=head_inflow,
                    interval_inflow=interval_inflow,
                    V0=V0,
                    V_target=terminal_effective_target,
                    q_min=qmin,
                    q_max=qmax,
                    V_min=joint_vmin,
                    V_max=joint_vmax,
                    delta_t=dt_vec,
                    flow_ramp_up=flow_ramps.get("ramp_up"),
                    flow_ramp_down=flow_ramps.get("ramp_down"),
                )
                q_in_pred = _compute_cascade_inflows(q_phys, head_inflow, interval_inflow)
                projection_state = {"V": joint_storage, "meta": None}
                model.last_projection = projection_state
                V0_use = storage_V0.expand(V0.shape[0], -1)
                V_target_use = V_target
                dt_vec_loss = dt_vec.to(dtype=q_phys.dtype, device=q_phys.device)
                Vmin_loss = Vmin_use.to(dtype=q_phys.dtype, device=q_phys.device)
                Vmax_loss = Vmax_use.to(dtype=q_phys.dtype, device=q_phys.device)
                raw_need_val.append(
                    _terminal_metrics_raw(
                        q_base_val.detach(),
                        q_in_obs.detach(),
                        V0.detach(),
                        dt_vec_loss,
                        V_target.detach(),
                    )[0]
                )
                power_value = None
                if power_energy_module is not None and criterion.w_power > 0.0:
                    try:
                        power_scores = power_energy_module(
                            q_out=q_phys,
                            q_in=q_in_pred,
                            V0=V0_use,
                            dt_steps=dt_vec_loss,
                        )
                        power_value = power_scores.mean()
                    except Exception as exc:
                        if global_step == 0:
                            print(f"[warn] power surrogate evaluation failed: {exc}")
                        power_value = None
                loss = criterion(
                    q_phys,
                    targets_phys,
                    physical_predictions=q_phys,
                    q_in=q_in_pred,
                    V0=V0_use,
                    V_target=V_target_use,
                    q_min=qmin,
                    q_max=qmax,
                    window_k=criterion.terminal_loss.window_k if criterion.terminal_loss is not None else None,
                    to_level=to_level_fn,
                    delta_t_steps=dt_vec_loss,
                    V_min=Vmin_loss,
                    V_max=Vmax_loss,
                    terminal_effective_target=terminal_effective_target,
                )
                if power_value is not None:
                    loss = loss - criterion.w_power * power_value
                consistency_loss = q_phys.new_tensor(0.0)
                q_consistent = _cascade_consistency_forward(
                    batch,
                    qmin,
                    qmax,
                    q_in_pred,
                    q_phys,
                    V0,
                    V_target,
                    dt_vec,
                    Vmin_use,
                    Vmax_use,
                    is_training=False,
                )
                if q_consistent is not None:
                    consistency_loss = torch.nn.functional.mse_loss(q_consistent, q_phys)
                    loss = loss + cascade_consistency_weight * consistency_loss
                batch_size_cur = features.size(0)
                val_loss_sum += loss.item() * batch_size_cur
                err_val = q_phys - targets_phys
                val_sq_err_sum += float((err_val * err_val).sum().item())
                val_abs_target_sum += float(targets_phys.abs().sum().item())
                val_elem_count += int(targets_phys.numel())
                if criterion.barrier_loss is not None and criterion.w_barrier:
                    barrier_value = criterion.barrier_loss(q_phys, qmin, qmax).item()
                    val_barrier_sum += barrier_value * batch_size_cur

                violation = ((q_phys < qmin) | (q_phys > qmax)).float().mean().item()
                dt_vol = dt_vec_loss.view(1, -1, 1) / 1e8
                dt_vol = dt_vol.expand(q_phys.size(0), -1, -1)
                projection_V = projection_state.get("V") if projection_state is not None else None
                if isinstance(projection_V, torch.Tensor):
                    storage_path = projection_V
                    if storage_path.dim() == 3 and storage_path.size(0) == 1 and q_phys.size(0) > 1:
                        storage_path = storage_path.expand(q_phys.size(0), -1, -1)
                else:
                    storage_path = V0.unsqueeze(1) + torch.cumsum((q_in_pred - q_phys) * dt_vol, dim=1)
                V_T = storage_path[:, -1, :]
                need = (terminal_effective_target - V_T).abs()
                K_use = max(0, min(window_k_eval, q_phys.size(1)))
                if K_use > 0:
                    cap = ((qmax[:, -K_use:, :] - qmin[:, -K_use:, :]) * dt_vol[:, -K_use:, :]).sum(dim=1)
                else:
                    cap = torch.zeros_like(need)
                shortfall = torch.clamp(need - cap, min=0.0)
                # Safe access to projection_state (may be None if hard_projection is disabled)
                infeasible_meta = None
                if projection_state is not None:
                    infeasible_meta = projection_state.get("meta", {}).get("infeasible") if isinstance(projection_state.get("meta"), dict) else None
                
                if isinstance(infeasible_meta, torch.Tensor):
                    infeasible_mask = infeasible_meta.to(dtype=torch.float32)
                    infeasible_num += infeasible_mask.sum().item()
                    total_targets += infeasible_mask.numel()
                else:
                    infeasible_num += (shortfall > 0).float().sum().item()
                    total_targets += shortfall.numel()
                need_values.extend(need.detach().cpu().view(-1).tolist())
                cap_values.extend(cap.detach().cpu().view(-1).tolist())
                term_err = need.mean().item()
                ramp = ((q_phys[:, 1:] - q_phys[:, :-1]) ** 2).mean().item()
                power_val = power_value.item() if power_value is not None else 0.0

                val_violation_sum += violation * batch_size_cur
                val_term_err_sum += term_err * batch_size_cur
                val_ramp_sum += ramp * batch_size_cur
                val_power_sum += power_val * batch_size_cur
                val_consistency_sum += consistency_loss.item() * batch_size_cur
                val_count += batch_size_cur

                val_preds_epoch.append(q_phys.detach().cpu().numpy())
                val_targets_epoch.append(targets_phys.detach().cpu().numpy())

        val_dataset_size = max(1, len(val_loader.dataset))
        val_loss = val_loss_sum / val_dataset_size
        val_rmse = math.sqrt(val_sq_err_sum / max(1, val_elem_count))
        val_mean_abs_target = val_abs_target_sum / max(1, val_elem_count)
        val_nrmse = val_rmse / max(1e-8, val_mean_abs_target)
        avg_violation = val_violation_sum / max(1, val_count)
        avg_term_err = val_term_err_sum / max(1, val_count)
        avg_ramp = val_ramp_sum / max(1, val_count)
        avg_power = val_power_sum / max(1, val_count)
        avg_barrier = val_barrier_sum / max(1, val_count)
        avg_consistency = val_consistency_sum / max(1, val_count)
        avg_infeasible = infeasible_num / total_targets if total_targets else 0.0
        median_need = float(np.median(need_values)) if need_values else 0.0
        median_cap = float(np.median(cap_values)) if cap_values else 0.0
        raw_val_need_mean = float(np.nanmean(raw_need_val)) if raw_need_val else float("nan")

        if selection_mode == "power_first":
            violation_excess = max(0.0, avg_violation - feasible_violation_tol)
            infeasible_excess = max(0.0, avg_infeasible - feasible_infeasible_tol)
            term_excess = max(0.0, avg_term_err - feasible_term_tol)
            composite_score = (
                (avg_power * power_coef)
                - (avg_ramp * ramp_coef)
                - (term_excess * term_coef)
                - (violation_excess * violation_coef)
                - (infeasible_excess * infeasible_penalty)
            )
        else:
            composite_score = (
                (-avg_term_err * term_coef)
                - (avg_violation * violation_coef)
                - (avg_ramp * ramp_coef)
                + (avg_power * power_coef)
            )
        if train_loss_ref is None or not np.isfinite(train_loss_ref) or train_loss_ref <= 0.0:
            train_loss_ref = max(abs(train_loss), 1e-12)
        if val_loss_ref is None or not np.isfinite(val_loss_ref) or val_loss_ref <= 0.0:
            val_loss_ref = max(abs(val_loss), 1e-12)
        train_loss_norm = abs(train_loss) / train_loss_ref
        val_loss_norm = abs(val_loss) / val_loss_ref
        loss_history_rows.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_loss_norm": float(train_loss_norm),
                "val_loss_norm": float(val_loss_norm),
                "train_rmse": float(train_rmse),
                "val_rmse": float(val_rmse),
                "train_nrmse": float(train_nrmse),
                "val_nrmse": float(val_nrmse),
                "term_err": float(avg_term_err),
                "violation": float(avg_violation),
                "infeasible_ratio": float(avg_infeasible),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        scheduler.step(val_loss)

        if composite_score > best_score + 1e-8:
            best_score = composite_score
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            if ms_tcn is not None:
                best_ms_state = copy.deepcopy(ms_tcn.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement on feasibility score for {patience} epochs).")
                break

        warn_marker = ""
        if infeasible_warn_ratio > 0.0 and avg_infeasible > infeasible_warn_ratio:
            warn_marker = " ?"

        if log_mode == "detailed":
            print(
                f"[Epoch {epoch:03d}] train={train_loss:.5f} val={val_loss:.5f} "
                f"term_err={avg_term_err:.5f} violation={avg_violation:.6f} ramp={avg_ramp:.5f} "
                f"power={avg_power:.5f} barrier={avg_barrier:.5f} cascade={avg_consistency:.5f} score={composite_score:.5f} "
                f"w_flow={criterion.w_flow:.3f} w_term={criterion.w_term:.3f} w_power={criterion.w_power:.3f} "
                f"K={criterion.terminal_loss.window_k if criterion.terminal_loss else 0} "
                f"infeas={avg_infeasible:.4f}{warn_marker} need_med={median_need:.3e} cap_med={median_cap:.3e} "
                f"rawNeed(train)={raw_need_epoch_mean:.3f} rawNeed(val)={raw_val_need_mean:.3f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
        else:
            print(
                f"[Epoch {epoch:03d}] train_loss={train_loss_norm:.5f} val_loss={val_loss_norm:.5f} "
                f"train_nrmse={train_nrmse:.5f} val_nrmse={val_nrmse:.5f} "
                f"term_err={avg_term_err:.5f} infeas={avg_infeasible:.4f}{warn_marker} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        val_preds = val_preds_epoch
        val_targets = val_targets_epoch

    model.load_state_dict(best_state)
    if ms_tcn is not None and best_ms_state is not None:
        ms_tcn.load_state_dict(best_ms_state)
    if ms_tcn is not None:
        ms_tcn.eval()
    model.eval()

    # ------------------------------------------------------------------
    # Final validation metrics
    # ------------------------------------------------------------------
    preds = np.concatenate(val_preds, axis=0) if val_preds else np.empty((0, sequence_length, output_dim))
    targets = np.concatenate(val_targets, axis=0) if val_targets else np.empty((0, sequence_length, output_dim))
    if preds.size and targets.size:
        y_true_t = torch.from_numpy(targets)
        y_pred_t = torch.from_numpy(preds)
        avg_metrics = calculate_performance_metrics(y_true_t, y_pred_t)
    else:
        avg_metrics = {"mse": 0.0, "rmse": 0.0, "mae": 0.0}
    reservoir_metrics = {"overall": avg_metrics}

    # ------------------------------------------------------------------
    # Persist artifacts
    # ------------------------------------------------------------------
    results_dir = script_dir / data_cfg.get("results_dir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    model_name = best_params.get("model_name", "multiscale")
    model_path = results_dir / f"{model_name}_best_model.pth"
    checkpoint_payload = {"model_state_dict": model.state_dict(), "hyperparameters": best_params}
    if ms_tcn is not None:
        checkpoint_payload["multiscale_tcn_state_dict"] = ms_tcn.state_dict()
    torch.save(checkpoint_payload, model_path)

    params_out = dict(best_params)
    params_out.update(
        {
            "input_dim": augmented_input_dim,
            "output_dim": output_dim,
            "sequence_length": sequence_length,
            "output_sequence_length": output_sequence_length,
            "dropout": dropout,
            "enable_sequence_decoding": enable_sequence_decoding,
            "enable_multiscale": enable_multiscale,
            "train_years": train_years,
            "validation_years": val_years,
        }
    )
    params_out.setdefault("model_type", "multiscale")
    with open(results_dir / f"{model_name}_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(params_out, f, ensure_ascii=False, indent=2)

    transforms_path = results_dir / "annual_transforms.pkl"
    train_dataset.save_transforms(str(transforms_path))

    if ms_tcn is not None:
        setattr(model, "multiscale_tcn", ms_tcn)

    history_csv_path = results_dir / loss_curve_csv
    history_png_path = results_dir / loss_curve_png
    if loss_history_rows:
        history_df = pd.DataFrame(loss_history_rows)
        history_df.to_csv(history_csv_path, index=False, encoding="utf-8-sig")
        try:
            fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
            ax.plot(history_df["epoch"], history_df["train_loss_norm"], label="Training Loss", linewidth=2.2)
            ax.plot(history_df["epoch"], history_df["val_loss_norm"], label="Validation Loss", linewidth=2.2)
            ax.set_xlabel("Training Epoch")
            ax.set_ylabel("Normalized Loss")
            ax.grid(True, linestyle="--", alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(history_png_path, dpi=300)
            plt.close(fig)
        except Exception as exc:
            print(f"[warn] failed to save loss curve image: {exc}")

    print("Training finished")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Model saved to: {model_path}")
    print(f"  Transforms saved to: {transforms_path}")
    if loss_history_rows:
        print(f"  Loss history csv: {history_csv_path}")
        print(f"  Loss curve png: {history_png_path}")

    return model, avg_metrics["r2"], avg_metrics["corr"], avg_metrics["rmse"], avg_metrics["mape"]

# ---------------------------------------------------------------------------
# Optional configuration loader
# ---------------------------------------------------------------------------
try:
    from config_loader import get_config, cfg_lookup, cfg_lookup_bool, parse_window

    CONFIG_AVAILABLE = True
    print("Configuration loader available")
except ImportError:
    CONFIG_AVAILABLE = False
    get_config = None  # type: ignore
    print("Configuration loader not found - using built-in defaults")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
if not CONFIG_AVAILABLE:
    # Define no-op config helpers when config_loader is unavailable
    def cfg_lookup(key: str, default: Any = None, cfg: Any = None) -> Any:  # type: ignore
        return default
    def cfg_lookup_bool(key: str, default: bool = False, cfg: Any = None) -> bool:  # type: ignore
        return bool(default)
    def parse_window(value: Any, fallback: int, total: int) -> int:  # type: ignore
        return fallback
def _clean_column(name: str) -> str:
    """Normalise column names by removing units and whitespace."""

    if not isinstance(name, str):
        return name

    cleaned = name.strip()
    for token in ("(", ")", "（", "）", " "):
        cleaned = cleaned.replace(token, "")

    for token in ("m3/s", "m^3/s", "M3/S", "m3/s", "m3/S"):
        cleaned = cleaned.replace(token, "")

    cleaned = cleaned.replace("_区间来水", "区间来水")
    cleaned = cleaned.replace("区间来水", "_区间来水")
    return cleaned


def _set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set deterministic seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False



def _lerp(start: float, end: float, ratio: float) -> float:
    """Linear interpolate between start and end."""

    return float(start + (end - start) * ratio)


def _compute_cascade_inflows(
    q_phys: torch.Tensor,
    head_inflow: torch.Tensor,
    interval_inflow: Optional[torch.Tensor],
) -> torch.Tensor:
    """Compute reservoir inflows accounting for upstream releases."""

    B, T, R = q_phys.shape
    q_in = torch.zeros_like(q_phys)
    if head_inflow.ndim == 3:
        q_in[:, :, 0] = head_inflow[..., 0]
    else:
        q_in[:, :, 0] = head_inflow

    if interval_inflow is None:
        interval_inflow = q_phys.new_zeros((B, T, max(0, R - 1)))
    elif interval_inflow.ndim == 2:
        interval_inflow = interval_inflow.unsqueeze(-1)

    num_interval = interval_inflow.size(-1)
    for r in range(1, R):
        interval_slice = interval_inflow[:, :, r - 1] if (r - 1) < num_interval else 0.0
        if not torch.is_tensor(interval_slice):
            interval_slice = torch.tensor(interval_slice, dtype=q_phys.dtype, device=q_phys.device)
        q_in[:, :, r] = q_phys[:, :, r - 1] + interval_slice
    return q_in


def _resolve_self_consistent_iters(
    cfg_obj: Any,
    section: str,
    default_iters: int = SELF_CONSISTENT_ITERS_DEFAULT,
    max_iters: int = 6,
) -> int:
    """Read self-consistent iteration count from config section with safe bounds."""

    try:
        if hasattr(cfg_obj, "get"):
            sec_cfg = (cfg_obj.get(section, {}) or {})
        elif isinstance(cfg_obj, dict):
            sec_cfg = (cfg_obj.get(section, {}) or {})
        else:
            sec_cfg = {}
        val = int(sec_cfg.get("self_consistent_iters", default_iters))
    except Exception:
        val = int(default_iters)
    return max(1, min(int(val), int(max_iters)))


def _run_self_consistent_refinement(
    *,
    q_phys: torch.Tensor,
    raw_logits: Optional[torch.Tensor],
    model: torch.nn.Module,
    head_inflow: torch.Tensor,
    interval_inflow: Optional[torch.Tensor],
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    V0: torch.Tensor,
    V_target: torch.Tensor,
    delta_t: torch.Tensor,
    V_min: torch.Tensor,
    V_max: torch.Tensor,
    sc_iters: int,
    post_step: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Shared self-consistent inflow loop used by train/val/generation."""

    q_prev = None
    post_payload: Dict[str, Any] = {}
    q_cur = q_phys
    for _ in range(max(1, int(sc_iters))):
        q_in_pred = _compute_cascade_inflows(q_cur, head_inflow, interval_inflow)
        q_rebuilt = _rebuild_capacity_flow(
            model=model,
            raw_logits=raw_logits,
            q_min=q_min,
            q_max=q_max,
            q_in=q_in_pred,
            V0=V0,
            V_target=V_target,
            delta_t=delta_t,
            V_min=V_min,
            V_max=V_max,
        )
        if q_rebuilt is not None:
            q_cur = q_rebuilt
            q_in_pred = _compute_cascade_inflows(q_cur, head_inflow, interval_inflow)

        if post_step is not None:
            try:
                q_post, payload = post_step(q_cur, q_in_pred)
                if isinstance(q_post, torch.Tensor):
                    q_cur = q_post
                if isinstance(payload, dict):
                    post_payload = payload
            except Exception:
                pass

        if q_prev is not None:
            diff = (q_cur - q_prev).abs().max().item()
            if diff <= SELF_CONSISTENT_TOL:
                break
        q_prev = q_cur.detach()

    q_in_pred = _compute_cascade_inflows(q_cur, head_inflow, interval_inflow)
    return {
        "q_phys": q_cur,
        "q_in_pred": q_in_pred,
        "payload": post_payload,
    }


def _update_post_projection_terminal_policy(
    *,
    req_residual: torch.Tensor,
    V_terminal: torch.Tensor,
    initial_reachable_mask: torch.Tensor,
    current_reachable_mask: torch.Tensor,
    current_best_effort_target: torch.Tensor,
    reservoir_names: Sequence[str],
    tol: float,
    allow_fallback: bool,
) -> Dict[str, Any]:
    """Apply post-projection terminal policy update with optional fallback."""

    if req_residual.dim() == 1:
        req_residual = req_residual.unsqueeze(0)
    if V_terminal.dim() == 1:
        V_terminal = V_terminal.unsqueeze(0)
    if initial_reachable_mask.dim() == 1:
        initial_reachable_mask = initial_reachable_mask.unsqueeze(0)
    if current_reachable_mask.dim() == 1:
        current_reachable_mask = current_reachable_mask.unsqueeze(0)
    if current_best_effort_target.dim() == 1:
        current_best_effort_target = current_best_effort_target.unsqueeze(0)

    hard_fail_mask = initial_reachable_mask & (req_residual.abs() > float(tol))
    hard_gap: Dict[str, float] = {}
    fallback_gap: Dict[str, float] = {}
    msg_parts: List[str] = []
    fail_idx = hard_fail_mask.squeeze(0).nonzero(as_tuple=False).view(-1).tolist()
    for ridx in fail_idx:
        nm = reservoir_names[ridx] if ridx < len(reservoir_names) else str(ridx)
        gap_val = float(req_residual.abs()[0, ridx].item())
        hard_gap[nm] = max(hard_gap.get(nm, 0.0), gap_val)
        if allow_fallback:
            fallback_gap[nm] = max(fallback_gap.get(nm, 0.0), gap_val)
        msg_parts.append(f"{nm} gap={gap_val:.3f}e8m3")

    next_mask = current_reachable_mask
    next_target = current_best_effort_target
    if allow_fallback and torch.any(hard_fail_mask):
        next_mask = current_reachable_mask & (~hard_fail_mask)
        next_target = torch.where(hard_fail_mask, V_terminal.detach(), current_best_effort_target)

    return {
        "hard_fail_mask": hard_fail_mask,
        "reachable_mask": next_mask,
        "best_effort_target": next_target,
        "hard_gap": hard_gap,
        "fallback_gap": fallback_gap,
        "message_parts": msg_parts,
    }


def _build_generation_stats(
    *,
    violation_rate: float,
    terminal_error: float,
    infeasible_ratio: float,
    reach_report: Dict[str, Any],
    allow_post_projection_fallback: bool,
    post_hard_gap: Dict[str, float],
) -> Dict[str, Any]:
    """Build a stable generation stats schema for downstream reports."""

    return {
        "schema_version": "v2",
        "violation_rate": float(violation_rate),
        "terminal_error": float(terminal_error),
        "infeasible_ratio": float(infeasible_ratio),
        "precheck_gap_max": float(reach_report.get("max_gap", 0.0)),
        "precheck_gap_avg": float(reach_report.get("avg_gap", 0.0)),
        "post_projection_fallback_enabled": bool(allow_post_projection_fallback),
        "post_projection_hard_gap_max": float(max(post_hard_gap.values())) if post_hard_gap else 0.0,
        "post_projection_hard_gap": {str(k): float(v) for k, v in post_hard_gap.items()},
        "terminal_reachability": reach_report,
    }


def _compute_terminal_reachability(
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    head_inflow: torch.Tensor,
    interval_inflow: Optional[torch.Tensor],
    V0: torch.Tensor,
    delta_t: torch.Tensor,
    V_min: torch.Tensor,
    V_max: torch.Tensor,
    V_target: torch.Tensor,
    V_T_lo: Optional[torch.Tensor] = None,
    V_T_hi: Optional[torch.Tensor] = None,
    reservoir_names: Optional[Sequence[str]] = None,
    tol: float = 1.0e-6,
) -> Dict[str, Any]:
    """Compute a joint terminal reachability envelope under hard box constraints."""

    device = q_min.device
    dtype = q_min.dtype
    B, T, R = q_min.shape

    if head_inflow.dim() == 3:
        head_flow = head_inflow[..., 0]
    else:
        head_flow = head_inflow
    head_flow = head_flow.to(device=device, dtype=dtype)

    if interval_inflow is None:
        interval_flow = q_min.new_zeros((B, T, max(0, R - 1)))
    else:
        interval_flow = interval_inflow.to(device=device, dtype=dtype)
        if interval_flow.dim() == 2:
            interval_flow = interval_flow.unsqueeze(-1)

    dt_use = delta_t.to(device=device, dtype=dtype)
    if dt_use.dim() == 1:
        dt_use = dt_use.view(1, -1, 1).expand(B, -1, R)
    elif dt_use.dim() == 2:
        dt_use = dt_use.unsqueeze(-1).expand(B, -1, R)
    elif dt_use.dim() == 3:
        dt_use = dt_use.expand(B, -1, R)
    else:
        raise ValueError("delta_t must have shape [T], [B,T], or [B,T,R]")
    dt_vol = dt_use / 1e8

    V0_use = V0.to(device=device, dtype=dtype)
    if V0_use.dim() == 1:
        V0_use = V0_use.unsqueeze(0).expand(B, -1)
    elif V0_use.size(0) == 1 and B > 1:
        V0_use = V0_use.expand(B, -1)

    V_target_use = V_target.to(device=device, dtype=dtype)
    if V_target_use.dim() == 1:
        V_target_use = V_target_use.unsqueeze(0).expand(B, -1)
    elif V_target_use.size(0) == 1 and B > 1:
        V_target_use = V_target_use.expand(B, -1)

    if V_min.dim() == 2:
        V_min = V_min.unsqueeze(0).expand(B, -1, -1)
    if V_max.dim() == 2:
        V_max = V_max.unsqueeze(0).expand(B, -1, -1)
    V_min = V_min.to(device=device, dtype=dtype)
    V_max = V_max.to(device=device, dtype=dtype)

    def _expand_terminal_bound(bound: Optional[torch.Tensor], fallback: torch.Tensor) -> torch.Tensor:
        if bound is None:
            return fallback.clone()
        bound_use = bound.to(device=device, dtype=dtype)
        if bound_use.dim() == 1:
            bound_use = bound_use.unsqueeze(0).expand(B, -1)
        elif bound_use.dim() == 2 and bound_use.size(0) == 1 and B > 1:
            bound_use = bound_use.expand(B, -1)
        return bound_use

    target_lo = _expand_terminal_bound(V_T_lo, V_target_use)
    target_hi = _expand_terminal_bound(V_T_hi, V_target_use)

    def _simulate_terminal_bounds(maximize_storage: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        V_cur = V0_use.clone()
        V_path: List[torch.Tensor] = []
        q_path: List[torch.Tensor] = []
        for t in range(T):
            q_release_t = q_min.new_zeros((B, R))
            V_next = q_min.new_zeros((B, R))
            q_in_t = q_min.new_zeros((B, R))
            q_in_t[:, 0] = head_flow[:, t]
            for r in range(R):
                if r > 0:
                    inter = interval_flow[:, t, r - 1] if interval_flow.size(-1) >= r else q_min.new_zeros((B,))
                    q_in_t[:, r] = q_release_t[:, r - 1] + inter
                k_t = torch.clamp(dt_vol[:, t, r], min=1.0e-12)
                v_cur_r = V_cur[:, r]
                v_lo_r = V_min[:, t, r]
                v_hi_r = V_max[:, t, r]
                q_lo_r = q_min[:, t, r]
                q_hi_r = q_max[:, t, r]
                q_in_r = q_in_t[:, r]
                if maximize_storage:
                    required_release = q_in_r - (v_hi_r - v_cur_r) / k_t
                    q_rel_r = torch.maximum(q_lo_r, required_release)
                    q_rel_r = torch.minimum(q_rel_r, q_hi_r)
                else:
                    allowed_release = q_in_r - (v_lo_r - v_cur_r) / k_t
                    q_rel_r = torch.minimum(q_hi_r, allowed_release)
                    q_rel_r = torch.maximum(q_rel_r, q_lo_r)
                v_next_r = v_cur_r + (q_in_r - q_rel_r) * k_t
                v_next_r = torch.minimum(torch.maximum(v_next_r, v_lo_r), v_hi_r)
                q_release_t[:, r] = q_rel_r
                V_next[:, r] = v_next_r
            q_path.append(q_release_t)
            V_path.append(V_next)
            V_cur = V_next
        return torch.stack(V_path, dim=1), torch.stack(q_path, dim=1)

    V_path_max, q_path_max = _simulate_terminal_bounds(maximize_storage=True)
    V_path_min, q_path_min = _simulate_terminal_bounds(maximize_storage=False)
    max_reachable = V_path_max[:, -1, :]
    min_reachable = V_path_min[:, -1, :]

    gap_lo = torch.clamp(target_lo - max_reachable, min=0.0)
    gap_hi = torch.clamp(min_reachable - target_hi, min=0.0)
    gap_to_target = torch.maximum(gap_lo, gap_hi)
    reachable_mask = gap_to_target <= tol

    names = list(reservoir_names) if reservoir_names is not None else [str(i) for i in range(R)]
    per_reservoir: Dict[str, Dict[str, Any]] = {}
    for r in range(R):
        name = names[r] if r < len(names) else str(r)
        gap_lo_r = gap_lo[:, r]
        gap_hi_r = gap_hi[:, r]
        gap_target_r = gap_to_target[:, r]
        reachable_r = reachable_mask[:, r]
        status = "reachable"
        if bool(torch.any(gap_lo_r > tol).item()):
            status = "insufficient_upper_reach"
        elif bool(torch.any(gap_hi_r > tol).item()):
            status = "excess_lower_reach"
        per_reservoir[name] = {
            "reachable": bool(torch.all(reachable_r).item()),
            "reachable_ratio": float(reachable_r.float().mean().item()),
            "status": status,
            "target_nominal": float(V_target_use[:, r].mean().item()),
            "target_lower": float(target_lo[:, r].mean().item()),
            "target_upper": float(target_hi[:, r].mean().item()),
            "min_reachable_terminal": float(min_reachable[:, r].min().item()),
            "max_reachable_terminal": float(max_reachable[:, r].max().item()),
            "gap_to_target": float(gap_target_r.max().item()),
            "gap_to_lower_bound": float(gap_lo_r.max().item()),
            "gap_to_upper_bound": float(gap_hi_r.max().item()),
        }

    unreachable = [name for name, info in per_reservoir.items() if not info["reachable"]]
    return {
        "reachable_mask": reachable_mask,
        "gap_to_target": gap_to_target,
        "min_reachable_terminal": min_reachable,
        "max_reachable_terminal": max_reachable,
        "target_lower": target_lo,
        "target_upper": target_hi,
        "target_nominal": V_target_use,
        "max_storage_path": V_path_max,
        "min_storage_path": V_path_min,
        "max_release_path": q_path_max,
        "min_release_path": q_path_min,
        "report": {
            "enabled": True,
            "reachable_ratio": float(reachable_mask.float().mean().item()),
            "hard_terminal_enforced_ratio": float(reachable_mask.float().mean().item()),
            "unreachable_reservoirs": unreachable,
            "max_gap": float(gap_to_target.max().item()),
            "avg_gap": float(gap_to_target.mean().item()),
            "per_reservoir": per_reservoir,
        },
    }


def _resolve_terminal_policy_targets(
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    head_inflow: torch.Tensor,
    interval_inflow: Optional[torch.Tensor],
    V0: torch.Tensor,
    delta_t: torch.Tensor,
    V_min: torch.Tensor,
    V_max: torch.Tensor,
    V_target: torch.Tensor,
    V_T_lo: Optional[torch.Tensor] = None,
    V_T_hi: Optional[torch.Tensor] = None,
    reservoir_names: Optional[Sequence[str]] = None,
    tol: float = 1.0e-6,
) -> Dict[str, Any]:
    reachability = _compute_terminal_reachability(
        q_min=q_min,
        q_max=q_max,
        head_inflow=head_inflow,
        interval_inflow=interval_inflow,
        V0=V0,
        delta_t=delta_t,
        V_min=V_min,
        V_max=V_max,
        V_target=V_target,
        V_T_lo=V_T_lo,
        V_T_hi=V_T_hi,
        reservoir_names=reservoir_names,
        tol=tol,
    )
    target_dtype = V_target.dtype
    target_device = V_target.device
    effective_target = torch.maximum(
        reachability["min_reachable_terminal"].to(device=target_device, dtype=target_dtype),
        torch.minimum(
            V_target.to(device=target_device, dtype=target_dtype),
            reachability["max_reachable_terminal"].to(device=target_device, dtype=target_dtype),
        ),
    )
    return {
        "reachability": reachability,
        "reachable_mask": reachability["reachable_mask"].to(device=target_device),
        "effective_target": effective_target,
    }


def _rebuild_capacity_flow(
    model: torch.nn.Module,
    raw_logits: Optional[torch.Tensor],
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    q_in: torch.Tensor,
    V0: torch.Tensor,
    V_target: torch.Tensor,
    delta_t: torch.Tensor,
    V_min: Optional[torch.Tensor] = None,
    V_max: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Recompute flow allocation using updated cascade inflows without re-encoding features."""

    if raw_logits is None or not hasattr(model, "capacity_head"):
        return None
    try:
        return model.capacity_head(raw_logits, q_min, q_max, q_in, V0, V_target, delta_t, V_min=V_min, V_max=V_max)
    except Exception:
        return None



# ---------------------------------------------------------------------------
# Simplified schedule generation (fallback implementation)
# ---------------------------------------------------------------------------
def generate_diverse_annual_schedules(
    model: torch.nn.Module,
    test_input: torch.Tensor,
    population_size: int = 50,
    year: int = 2024,
    output_dir: str = "schedule_results",
    device: str = "cpu",
    dataset: Optional[AnnualReservoirDataset] = None,
) -> Tuple[List[np.ndarray], Dict[str, float]]:
    """Decode schedules via the same differentiable path used during training."""

    if dataset is None:
        raise ValueError("dataset must be provided to supply physical constraints.")

    cfg = get_config() if CONFIG_AVAILABLE and get_config is not None else {}

    def _cfg_get(path: str, default: Any) -> Any:
        if not cfg:
            return default
        current: Any = cfg
        parts = path.split(".")
        for idx, part in enumerate(parts):
            last = idx == len(parts) - 1
            try:
                if hasattr(current, "get"):
                    current = current.get(part, default if last else {})
                elif isinstance(current, dict):
                    current = current.get(part, default if last else {})
                else:
                    current = getattr(current, part)
            except Exception:
                return default
            if current is None:
                return default if last else {}
        return current

    window_k = int(_cfg_get("loss.terminal.window_k", 10))
    multiscale_enabled = bool(_cfg_get("multiscale.enabled", False))
    multiscale_cfg = _cfg_get("multiscale", {})
    if hasattr(multiscale_cfg, "get"):
        tcn_cfg = multiscale_cfg.get("tcn", {})
    elif isinstance(multiscale_cfg, dict):
        tcn_cfg = multiscale_cfg.get("tcn", {})
    else:
        tcn_cfg = getattr(multiscale_cfg, "tcn", {})
    if hasattr(tcn_cfg, "get"):
        tcn_enabled = bool(tcn_cfg.get("enabled", False))
    elif isinstance(tcn_cfg, dict):
        tcn_enabled = bool(tcn_cfg.get("enabled", False))
    else:
        tcn_enabled = bool(getattr(tcn_cfg, "enabled", False))
    if isinstance(tcn_cfg, dict):
        tcn_channels = int(tcn_cfg.get("channels", 64))
        tcn_kernels = tuple(int(k) for k in tcn_cfg.get("kernels", [3, 5, 7]))
        tcn_dilations = tuple(int(d) for d in tcn_cfg.get("dilations", [1, 2, 3]))
    else:
        tcn_channels = int(getattr(tcn_cfg, "channels", 64))
        tcn_kernels = tuple(int(k) for k in getattr(tcn_cfg, "kernels", [3, 5, 7]))
        tcn_dilations = tuple(int(d) for d in getattr(tcn_cfg, "dilations", [1, 2, 3]))

    torch_device = torch.device(device)
    model = model.to(torch_device)
    ms_tcn = getattr(model, "multiscale_tcn", None)
    if ms_tcn is not None:
        ms_tcn = ms_tcn.to(torch_device)

    gen_mc_cfg = (cfg.get("generation", {}) or {}).get("mc_dropout", {}) if cfg else {}
    gen_terminal_cfg = (cfg.get("generation", {}) or {}).get("terminal_policy", {}) if cfg else {}
    allow_post_projection_fallback = bool(gen_terminal_cfg.get("allow_post_projection_fallback", False))
    use_mc_dropout = bool(gen_mc_cfg.get("enabled", False))
    if use_mc_dropout:
        model.train()
        if ms_tcn is not None:
            ms_tcn.train()
    else:
        model.eval()
        if ms_tcn is not None:
            ms_tcn.eval()
    features = test_input.to(torch_device)

    sample = dataset[-1]
    qmin = sample["q_min"].unsqueeze(0).to(torch_device)
    qmax = sample["q_max"].unsqueeze(0).to(torch_device)
    q_in_obs = sample["q_in"].unsqueeze(0).to(torch_device)
    head_inflow = sample["head_inflow"].unsqueeze(0).to(torch_device)
    interval_inflow_tensor = sample.get("interval_inflow")
    if interval_inflow_tensor is not None:
        interval_inflow = interval_inflow_tensor.unsqueeze(0).to(torch_device)
    else:
        interval_inflow = None
    # Provisional V0/VT from dataset; will be overridden by config if provided
    storage_V0 = sample["V0"].unsqueeze(0).to(torch_device)
    storage_VT = sample["V_target"].unsqueeze(0).to(torch_device)

    T = qmin.size(1)
    R = qmin.size(2)
    # Prefer config.yaml constraints.initial_storage/target_storage
    cfg_constraints = cfg.get("constraints", {}) if cfg else {}
    try:
        init_st = cfg_constraints.get("initial_storage", None) if hasattr(cfg_constraints, "get") else None
        tgt_st = cfg_constraints.get("target_storage", None) if hasattr(cfg_constraints, "get") else None
        # Only allow override when explicitly enabled in config
        allow_override = bool(cfg_constraints.get("allow_config_storage_override", False)) if hasattr(cfg_constraints, "get") else False
    except Exception:
        init_st, tgt_st = None, None
        allow_override = False
    if not allow_override:
        init_st, tgt_st = None, None
    if init_st is not None and tgt_st is not None:
        V0_list = list(map(float, init_st))
        VT_list = list(map(float, tgt_st))
        if len(V0_list) < R or len(VT_list) < R:
            raise RuntimeError("config.yaml constraints.initial_storage/target_storage ȲԸˮ")
        storage_V0 = torch.tensor(V0_list[:R], dtype=qmin.dtype, device=torch_device).unsqueeze(0)
        storage_VT = torch.tensor(VT_list[:R], dtype=qmin.dtype, device=torch_device).unsqueeze(0)
    dt_vec = torch.tensor(_TIME_STEP_SECONDS[:T], dtype=qmin.dtype, device=torch_device)
    dt_tensor = dt_vec.view(1, T, 1)
    dt_vol = dt_tensor / 1e8

    res_names = list(_RESERVOIR_NAMES or _DEFAULT_RESERVOIRS)[:R]
    constraints_csv = _SCRIPT_DIR / "shuxing" / "约束条件.csv"
    try:
        storage_Vmin, storage_Vmax, _, _ = load_storage_constraints_from_csv(
            constraints_csv,
            res_names,
            periods=T,
            device=torch_device,
        )
        constraints_df = pd.read_csv(str(constraints_csv), encoding="utf-8-sig")
    except Exception as exc:
        raise RuntimeError(f"加载库容约束失败: {exc}") from exc
    storage_Vmin = storage_Vmin.to(qmin.dtype).unsqueeze(0)
    storage_Vmax = storage_Vmax.to(qmin.dtype).unsqueeze(0)
    storage_V0 = storage_V0.to(qmin.dtype)
    storage_VT = storage_VT.to(qmin.dtype)
    storage_Vmin_policy = storage_Vmin
    storage_Vmax_policy = storage_Vmax

    proj_cfg = cfg.get("projection", {}) if cfg else {}
    use_level_projection = bool(proj_cfg.get("enabled", False))
    level_limits: Optional[Dict[str, torch.Tensor]] = None
    curves_for_proj: Optional[TorchCurves] = None
    reach_cfg = proj_cfg.get("reachability", {}) if isinstance(proj_cfg, dict) else {}
    terminal_reach_tol = 1e-4

    if use_level_projection:
        curves_for_proj = TorchCurves(
            str(_SHUXING_DIR / "curves"),
            str(_SHUXING_DIR),
            res_names,
        ).to(torch_device)
        curves_for_proj.eval()

        ramp_cfg = proj_cfg.get("ramp", {}) if isinstance(proj_cfg, dict) else {}
        ramp_source = str(ramp_cfg.get("source", "none")).lower()
        V_min_base = storage_Vmin.squeeze(0)
        V_max_base = storage_Vmax.squeeze(0)
        dyn_h_min = None
        dyn_h_max = None
        dyn_dh_up = None
        dyn_dh_dn = None
        try:
            _, _, dyn_constraints = load_dynamic_constraints_and_inflows(
                res_names,
                num_periods=T,
                inflow_year=int(year),
            )
            hmin_list: List[np.ndarray] = []
            hmax_list: List[np.ndarray] = []
            hup_list: List[np.ndarray] = []
            hdn_list: List[np.ndarray] = []
            for name in res_names:
                df_dyn = dyn_constraints.get(name)
                if df_dyn is None:
                    continue
                hmin_list.append(df_dyn["level_min"].to_numpy(dtype=np.float32)[:T])
                hmax_list.append(df_dyn["level_max"].to_numpy(dtype=np.float32)[:T])
                hup_list.append(df_dyn["level_up_max"].to_numpy(dtype=np.float32)[:T])
                hdn_list.append(df_dyn["level_down_max"].to_numpy(dtype=np.float32)[:T])
            if hmin_list and hmax_list:
                dyn_h_min = torch.tensor(
                    np.stack(hmin_list, axis=1),
                    dtype=qmin.dtype,
                    device=torch_device,
                )
                dyn_h_max = torch.tensor(
                    np.stack(hmax_list, axis=1),
                    dtype=qmin.dtype,
                    device=torch_device,
                )
                if hup_list and hdn_list:
                    dyn_dh_up = torch.tensor(
                        np.stack(hup_list, axis=1),
                        dtype=qmin.dtype,
                        device=torch_device,
                    )
                    dyn_dh_dn = torch.tensor(
                        np.stack(hdn_list, axis=1),
                        dtype=qmin.dtype,
                        device=torch_device,
                    )
                    # 如果 JSON 中成功加载了水位变幅约束而 ramp_source 仍为 none，
                    # 则自动启用配置来源，确保 make_level_limits 生效
                    if ramp_source == "none":
                        ramp_source = "config"
                        try:
                            print(
                                "[Generate] 已自动激活水位变幅约束 "
                                "(dH 来自 JSON), source set to 'config'"
                            )
                        except Exception:
                            pass
        except Exception as exc:
            print(f"[Generate] 警告: 动态约束加载失败 ({exc})，将忽略水位变幅约束。")
            dyn_h_min = None
            dyn_h_max = None
            dyn_dh_up = None
            dyn_dh_dn = None

        if ramp_source == "file":
            ramp_file = ramp_cfg.get("file")
            if ramp_file:
                ramp_path = Path(ramp_file)
                if not ramp_path.is_absolute():
                    ramp_path = _SCRIPT_DIR / ramp_path
                enc_opt = ramp_cfg.get("encoding")
                if isinstance(enc_opt, str):
                    encodings = [enc_opt]
                elif isinstance(enc_opt, (list, tuple)):
                    encodings = [str(enc) for enc in enc_opt]
                else:
                    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
                ramp_df = None
                last_exc: Optional[Exception] = None
                for enc in encodings:
                    try:
                        ramp_df = pd.read_csv(str(ramp_path), encoding=enc)
                        break
                    except Exception as exc:  # pragma: no cover - fallback path
                        last_exc = exc
                        continue
                if ramp_df is None:
                    print(f"[warn] ramp file {ramp_path} failed to load: {last_exc}")
                else:
                    ramp_df.columns = ramp_df.columns.astype(str).str.strip()
                    columns_cfg = ramp_cfg.get("columns", {}) if isinstance(ramp_cfg, dict) else {}

                    base_names_cfg = columns_cfg.get("base_names")
                    if isinstance(base_names_cfg, list) and len(base_names_cfg) == len(res_names):
                        base_refs = [str(name) for name in base_names_cfg]
                    else:
                        base_refs = list(res_names)

                    def _resolve_column_names(kind: str, default_suffix: str) -> List[str]:
                        cfg_val = columns_cfg.get(kind)
                        if isinstance(cfg_val, list):
                            if len(cfg_val) != len(base_refs):
                                raise ValueError(f"{kind} columns length mismatch: {len(cfg_val)} != {len(base_refs)}")
                            return [str(col) for col in cfg_val]
                        if isinstance(cfg_val, dict):
                            cols: List[str] = []
                            for ref in base_refs:
                                if ref not in cfg_val:
                                    raise KeyError(f"{kind} columns missing key {ref}")
                                cols.append(str(cfg_val[ref]))
                            return cols
                        suffix = None
                        if isinstance(cfg_val, str):
                            suffix = cfg_val
                        else:
                            suffix_cfg = columns_cfg.get(f"{kind}_suffix")
                            if isinstance(suffix_cfg, str):
                                suffix = suffix_cfg
                        if suffix is None:
                            suffix = default_suffix
                        return [f"{ref}{suffix}" for ref in base_refs]

                    def _stack_columns(col_names: List[str]) -> np.ndarray:
                        series_list: List[np.ndarray] = []
                        for col in col_names:
                            if col not in ramp_df.columns:
                                raise KeyError(f"ramp column '{col}' not found in {ramp_path}")
                            series = ramp_df[col].astype(float).to_numpy()
                            if series.size < T:
                                pad_val = series[-1] if series.size else 0.0
                                series = np.pad(series, (0, T - series.size), mode="edge")
                            series_list.append(series[:T])
                        return np.stack(series_list, axis=1)

                    try:
                        up_columns = _resolve_column_names("up", "水位升幅")
                        down_columns = _resolve_column_names("down", "水位降幅")
                        up_matrix = _stack_columns(up_columns)
                        down_matrix = _stack_columns(down_columns)
                    except Exception as exc:  # pragma: no cover - invalid config
                        print(f"[warn] ramp column resolution failed: {exc}")
                    else:
                        dyn_dh_up = torch.tensor(up_matrix, dtype=qmin.dtype, device=torch_device)
                        dyn_dh_dn = torch.tensor(down_matrix, dtype=qmin.dtype, device=torch_device)

        H_min_tensor, H_max_tensor, dH_up_tensor, dH_dn_tensor = make_level_limits(
            curves=curves_for_proj,
            V_min=V_min_base,
            V_max=V_max_base,
            H_min=dyn_h_min,
            H_max=dyn_h_max,
            ramp_source=ramp_source,
            ramp_up=dyn_dh_up,
            ramp_dn=dyn_dh_dn,
        )
        level_limits = {
            "H_min": H_min_tensor.to(qmin.dtype).to(torch_device),
            "H_max": H_max_tensor.to(qmin.dtype).to(torch_device),
            "dH_up": dH_up_tensor.to(qmin.dtype).to(torch_device),
            "dH_dn": dH_dn_tensor.to(qmin.dtype).to(torch_device),
        }
        # Tighten terminal reachability envelopes with dynamic level limits.
        # This avoids treating a target as "reachable" under storage boxes when
        # the dynamic H_min/H_max corridor actually makes it unattainable.
        try:
            if dyn_h_min is not None and dyn_h_max is not None and isinstance(curves_for_proj, TorchCurves):
                dyn_v_min = curves_for_proj.h2v_all(dyn_h_min).to(qmin.dtype).to(torch_device).unsqueeze(0)
                dyn_v_max = curves_for_proj.h2v_all(dyn_h_max).to(qmin.dtype).to(torch_device).unsqueeze(0)
                storage_Vmin_policy = torch.maximum(storage_Vmin, dyn_v_min)
                storage_Vmax_policy = torch.minimum(storage_Vmax, dyn_v_max)
        except Exception:
            storage_Vmin_policy = storage_Vmin
            storage_Vmax_policy = storage_Vmax
        # Prepare terminal storage window for level-projection guidance.
        # NOTE:
        # - V_T_lo/hi for projection can keep an h-tolerance corridor.
        # - Reachability gating for hard terminal enforcement must use near-exact
        #   targets so that "reachable => hard enforce" remains strict.
        V_T_lo_tensor: Optional[torch.Tensor] = None
        V_T_hi_tensor: Optional[torch.Tensor] = None
        V_T_lo_policy: Optional[torch.Tensor] = None
        V_T_hi_policy: Optional[torch.Tensor] = None
        try:
            term_cfg = (cfg.get("loss", {}) or {}).get("terminal", {}) or {}
            terminal_reach_tol = float(term_cfg.get("hard_reach_tolerance_volume", terminal_reach_tol))
            use_level_term = bool(term_cfg.get("use_level", False))
            tol_vol_policy = 1e-6
            V_T_lo_policy = (storage_VT - tol_vol_policy).to(qmin.dtype).to(torch_device)
            V_T_hi_policy = (storage_VT + tol_vol_policy).to(qmin.dtype).to(torch_device)
            if use_level_term and isinstance(curves_for_proj, TorchCurves):
                h_cfg = term_cfg.get("h_target", {}) or {}
                cfg_target_levels = (cfg.get("constraints", {}) or {}).get("target_levels", []) if cfg else []
                default_entry = h_cfg.get("default", {}) if isinstance(h_cfg, dict) else {}
                default_val = None
                if isinstance(default_entry, dict) and "value" in default_entry:
                    default_val = float(default_entry.get("value", 0.0))
                elif default_entry not in (None, {}):
                    try:
                        default_val = float(default_entry)
                    except (TypeError, ValueError):
                        default_val = None
                targets: List[float] = []
                for nm in res_names:
                    entry = h_cfg.get(nm, default_entry) if isinstance(h_cfg, dict) else default_entry
                    if isinstance(entry, dict):
                        value = entry.get("value", None)
                    elif entry not in (None, {}):
                        value = entry
                    else:
                        value = None
                    if value is None:
                        idx = res_names.index(nm)
                        if idx < len(cfg_target_levels):
                            value = cfg_target_levels[idx]
                        elif default_val is not None:
                            value = default_val
                        else:
                            value = None
                    try:
                        targets.append(float(value))
                    except (TypeError, ValueError):
                        targets = []
                        break
                if not targets:
                    tol_vol = 1e-6
                    V_T_lo_tensor = (storage_VT - tol_vol).to(qmin.dtype).to(torch_device)
                    V_T_hi_tensor = (storage_VT + tol_vol).to(qmin.dtype).to(torch_device)
                else:
                    H_target = torch.tensor(targets, dtype=qmin.dtype, device=torch_device)
                    h_tol = float(term_cfg.get("h_tolerance", 0.3))
                    V_T_lo_tensor = curves_for_proj.h2v_all(H_target - h_tol).unsqueeze(0)
                    V_T_hi_tensor = curves_for_proj.h2v_all(H_target + h_tol).unsqueeze(0)
            else:
                tol_vol = 1e-6  # Tighten tolerance for exact terminal matching
                V_T_lo_tensor = (storage_VT - tol_vol).to(qmin.dtype).to(torch_device)
                V_T_hi_tensor = (storage_VT + tol_vol).to(qmin.dtype).to(torch_device)
        except Exception:
            V_T_lo_tensor = None
            V_T_hi_tensor = None
            V_T_lo_policy = None
            V_T_hi_policy = None
    else:
        V_T_lo_tensor = None
        V_T_hi_tensor = None
        V_T_lo_policy = None
        V_T_hi_policy = None
    # Sanity: V0 must lie within first-period storage bounds
    v0_lo = storage_Vmin[:, 0, :]
    v0_hi = storage_Vmax[:, 0, :]
    viol = (storage_V0 < v0_lo) | (storage_V0 > v0_hi)
    if torch.any(viol):
        try:
            names = res_names
        except Exception:
            names = [str(i) for i in range(R)]
        bad = viol.squeeze(0).nonzero(as_tuple=False).view(-1).tolist()
        details = []
        for idx in bad:
            details.append(
                f"{names[idx]}: V0={float(storage_V0[0, idx]):.6f}, range=[{float(v0_lo[0, idx]):.6f},{float(v0_hi[0, idx]):.6f}]"
            )
        print("[generate] V0 out of bounds -> " + "; ".join(details))
        raise RuntimeError("ʼ(V0)ԼΧ config.yaml  Լ.csv Ƿһ")

    terminal_policy = _resolve_terminal_policy_targets(
        q_min=qmin,
        q_max=qmax,
        head_inflow=head_inflow,
        interval_inflow=interval_inflow,
        V0=storage_V0,
        delta_t=dt_vec,
        V_min=storage_Vmin_policy,
        V_max=storage_Vmax_policy,
        V_target=storage_VT,
        V_T_lo=V_T_lo_policy,
        V_T_hi=V_T_hi_policy,
        reservoir_names=res_names,
        tol=terminal_reach_tol,
    )
    terminal_reachability = terminal_policy["reachability"]
    terminal_reachable_mask = terminal_policy["reachable_mask"].to(device=device)
    terminal_best_effort_target = terminal_policy["effective_target"].to(device=device, dtype=storage_VT.dtype)
    initial_terminal_reachable_mask = terminal_reachable_mask.clone()
    reach_report = terminal_reachability["report"]
    if reach_report.get("unreachable_reservoirs"):
        msg_parts = []
        for name in reach_report["unreachable_reservoirs"]:
            info = reach_report["per_reservoir"].get(name, {})
            gap_val = float(info.get("gap_to_target", 0.0))
            status = str(info.get("status", "unreachable"))
            msg_parts.append(f"{name} gap={gap_val:.3f}e8m3 ({status})")
        if msg_parts:
            print(
                "[generate] year={} terminal target unreachable under hard constraints: {} "
                "(switch to best-effort terminal attainment)".format(int(year), ", ".join(msg_parts))
            )




    ms_inputs: Optional[Dict[str, torch.Tensor]] = None
    base_features = features

    def _compose_features_with_multiscale(
        base_feat: torch.Tensor,
        ms_tensor: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal ms_tcn
        ms_tensor = ms_tensor.to(torch_device)
        parts: List[torch.Tensor] = [base_feat, ms_tensor.reshape(ms_tensor.size(0), ms_tensor.size(1), -1)]
        if ms_tcn is None and tcn_enabled and getattr(model, "multiscale_tcn_state_dict", None) is not None:
            ms_tcn = MultiScaleTCN(
                in_dim=ms_tensor.size(-1),
                out_dim=tcn_channels,
                kernels=tcn_kernels,
                dilations=tcn_dilations,
            ).to(torch_device)
            ms_tcn.load_state_dict(getattr(model, "multiscale_tcn_state_dict"), strict=False)
            ms_tcn.eval()
            setattr(model, "multiscale_tcn", ms_tcn)
        if ms_tcn is not None:
            # Recreate TCN if channel count changed
            try:
                current_in = getattr(ms_tcn.branches[0][0], "in_channels", None)
                new_in = int(ms_tensor.size(-1))
                if current_in is not None and current_in != new_in:
                    ms_tcn = MultiScaleTCN(
                        in_dim=new_in,
                        out_dim=tcn_channels,
                        kernels=tcn_kernels,
                        dilations=tcn_dilations,
                    ).to(torch_device)
                    ms_tcn.eval()
            except Exception:
                pass
            ms_encoded, _ = ms_tcn(ms_tensor)
            parts.append(ms_encoded.reshape(ms_encoded.size(0), ms_encoded.size(1), -1))
        return torch.cat(parts, dim=-1)

    if multiscale_enabled:
        ms_inputs = {
            "q_in": q_in_obs,
            "q_min": qmin,
            "q_max": qmax,
            "delta_t": dt_vec,
            "V0": storage_V0,
            "V_target": storage_VT,
            "V_min": storage_Vmin.squeeze(0),
            "V_max": storage_Vmax.squeeze(0),
        }
        ms_stats = build_multiscale_features(ms_inputs, cfg)
        if ms_stats is not None:
            features = _compose_features_with_multiscale(base_features, ms_stats)

    proj = getattr(model, "shared_input_projection", None)
    expected_in_dim = getattr(proj, "in_features", None)
    if expected_in_dim is not None and features.size(-1) != expected_in_dim:
        if multiscale_enabled and ms_inputs is not None:
            # Backward compatibility: old checkpoints may not include terminal-guidance channels.
            ms_inputs_compat = {k: v for k, v in ms_inputs.items() if k not in {"delta_t", "V0", "V_target", "V_min", "V_max"}}
            ms_stats_compat = build_multiscale_features(ms_inputs_compat, cfg)
            if ms_stats_compat is not None:
                compat_features = _compose_features_with_multiscale(base_features, ms_stats_compat)
                if compat_features.size(-1) == expected_in_dim:
                    print(
                        "[generate] multiscale guidance disabled for checkpoint compatibility "
                        f"(feature_dim={features.size(-1)} -> {compat_features.size(-1)})."
                    )
                    features = compat_features
        if features.size(-1) == expected_in_dim:
            pass
        else:
            raise ValueError(
                f"Feature dimension mismatch: got {features.size(-1)}, expected {expected_in_dim}. "
                "Check multiscale feature configuration."
            )

    decoded: List[np.ndarray] = []
    violation_list: List[float] = []
    terminal_list: List[float] = []
    ramp_list: List[float] = []
    infeasible_list: List[float] = []
    need_values: List[float] = []
    cap_values: List[float] = []
    barrier_list: List[float] = []
    path_violation_list: List[float] = []
    barrier_loss_fn = (
        InteriorBarrierLoss(eps=_cfg_get("loss.barrier.eps", 1e-6))
        if bool(_cfg_get("loss.barrier.enabled", False))
        else None
    )

    dt_vec = dt_vec.to(q_in_obs.dtype)
    dt_tensor = dt_tensor.to(q_in_obs.dtype)
    storage_Vmin = storage_Vmin.to(q_in_obs.dtype)
    storage_Vmax = storage_Vmax.to(q_in_obs.dtype)
    storage_V0 = storage_V0.to(q_in_obs.dtype)
    storage_VT = storage_VT.to(q_in_obs.dtype)
    dt_vol = dt_tensor / 1e8

    residual_values: List[float] = []
    projection_infeasible: List[float] = []
    projected_paths: List[np.ndarray] = []
    post_unreachable: Dict[str, float] = {}
    post_hard_gap: Dict[str, float] = {}
    post_fallback_logged = False
    sample_count = max(1, int(population_size))
    sc_iters_generation = _resolve_self_consistent_iters(cfg, "generation", SELF_CONSISTENT_ITERS_DEFAULT)
    with torch.no_grad():
        for _ in range(sample_count):
            out = model(
                features,
                q_min=qmin,
                q_max=qmax,
                q_in=q_in_obs,
                V0=storage_V0,
                V_target=storage_VT,
                delta_t=dt_vec,
                V_min=storage_Vmin,
                V_max=storage_Vmax,
                terminal_reachable_mask=terminal_reachable_mask,
                terminal_best_effort_target=terminal_best_effort_target,
                return_logits=True,
            )
            if isinstance(out, tuple) and len(out) == 3:
                q_phys, raw_logits, _ = out
            else:
                q_phys = out if not isinstance(out, tuple) else out[0]
                raw_logits = None

            projection_V: Optional[torch.Tensor] = None
            meta: Dict[str, Any] = {}

            def _generation_post_step(q_cur: torch.Tensor, q_in_cur: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
                q_next = q_cur
                payload: Dict[str, Any] = {}
                if use_level_projection and level_limits is not None and curves_for_proj is not None:
                    q_iter = q_cur
                    hp_meta: Dict[str, Any] = {}
                    V_proj2 = None
                    for _ in range(2):
                        q_iter, level_qmin, level_qmax = _apply_level_projection(
                            q_raw=q_iter,
                            q_in=q_in_cur,
                            V0=storage_V0,
                            dt_seconds=dt_vec,
                            q_min=qmin,
                            q_max=qmax,
                            level_limits=level_limits,
                            curves=curves_for_proj,
                            V_T_lo=V_T_lo_tensor,
                            V_T_hi=V_T_hi_tensor,
                            reach_cfg=reach_cfg,
                        )
                        q_min_hp = torch.maximum(qmin, level_qmin)
                        q_max_hp = torch.minimum(qmax, level_qmax)
                        if hasattr(model, "hierarchical_projection"):
                            q_iter, V_proj2, meta2 = model.hierarchical_projection(
                                q_raw=q_iter,
                                q_in=q_in_cur,
                                V0=storage_V0,
                                V_target=storage_VT,
                                q_min=q_min_hp,
                                q_max=q_max_hp,
                                Vmin=storage_Vmin,
                                Vmax=storage_Vmax,
                                delta_t=dt_vec,
                                terminal_reachable_mask=terminal_reachable_mask,
                                terminal_best_effort_target=terminal_best_effort_target,
                            )
                            hp_meta = meta2
                    q_next = q_iter
                    if isinstance(V_proj2, torch.Tensor):
                        payload["projection_V"] = V_proj2
                    else:
                        dt_vol_expand = dt_vol.expand(q_next.size(0), -1, -1)
                        storage_base = storage_V0.expand(q_next.size(0), -1)
                        payload["projection_V"] = storage_base.unsqueeze(1) + torch.cumsum(
                            (q_in_cur - q_next) * dt_vol_expand, dim=1
                        )
                    payload["meta"] = hp_meta
                    return q_next, payload
                if use_level_projection and hasattr(model, "hierarchical_projection"):
                    try:
                        q_proj2, V_proj2, meta2 = model.hierarchical_projection(
                            q_raw=q_cur,
                            q_in=q_in_cur,
                            V0=storage_V0,
                            V_target=storage_VT,
                            q_min=qmin,
                            q_max=qmax,
                            Vmin=storage_Vmin,
                            Vmax=storage_Vmax,
                            delta_t=dt_vec,
                            terminal_reachable_mask=terminal_reachable_mask,
                            terminal_best_effort_target=terminal_best_effort_target,
                        )
                        payload["projection_V"] = V_proj2
                        payload["meta"] = meta2
                        return q_proj2, payload
                    except Exception:
                        return q_cur, {}
                return q_next, payload

            sc_result = _run_self_consistent_refinement(
                q_phys=q_phys,
                raw_logits=raw_logits,
                model=model,
                head_inflow=head_inflow,
                interval_inflow=interval_inflow,
                q_min=qmin,
                q_max=qmax,
                V0=storage_V0,
                V_target=terminal_best_effort_target,
                delta_t=dt_vec,
                V_min=storage_Vmin,
                V_max=storage_Vmax,
                sc_iters=sc_iters_generation,
                post_step=_generation_post_step,
            )
            q_phys = sc_result["q_phys"]
            q_in_pred = sc_result["q_in_pred"]
            payload = sc_result.get("payload", {})
            if isinstance(payload, dict):
                if isinstance(payload.get("projection_V"), torch.Tensor):
                    projection_V = payload["projection_V"]
                if isinstance(payload.get("meta"), dict):
                    meta = payload["meta"]
            if isinstance(projection_V, torch.Tensor):
                model.last_projection = {
                    "V": projection_V.detach(),
                    "meta": meta,
                }

            # Use the same differentiable joint layer as training after the
            # existing projections; the final cascade path is solved together.
            ramp_constraints = _CONSTRAINT_CFG.get("ramp_constraints", {})
            joint_vmin, joint_vmax = _joint_storage_bounds(
                storage_Vmin, storage_Vmax, level_limits, curves_for_proj
            )
            q_phys, projection_V = project_joint_schedule(
                q_proposed=q_phys,
                head_inflow=head_inflow,
                interval_inflow=interval_inflow,
                V0=storage_V0,
                V_target=terminal_best_effort_target,
                q_min=qmin,
                q_max=qmax,
                V_min=joint_vmin,
                V_max=joint_vmax,
                delta_t=dt_vec,
                flow_ramp_up=ramp_constraints.get("ramp_up"),
                flow_ramp_down=ramp_constraints.get("ramp_down"),
            )
            meta = {
                "residual": terminal_best_effort_target - projection_V[:, -1, :],
                "requested_residual": storage_VT - projection_V[:, -1, :],
                "infeasible": torch.zeros_like(terminal_best_effort_target, dtype=torch.bool),
            }
            model.last_projection = {"V": projection_V.detach(), "meta": meta}
            q_in_pred = _compute_cascade_inflows(q_phys, head_inflow, interval_inflow)
            decoded.append(q_phys.squeeze(0).detach().cpu().numpy())

            violation = ((q_phys < qmin) | (q_phys > qmax)).float().mean().item()
            storage_delta = (q_in_pred - q_phys) * dt_tensor / 1e8
            if isinstance(projection_V, torch.Tensor):
                storage_path = projection_V.to(q_phys.dtype)
                # Cache projected path for export
                projected_paths.append(storage_path.squeeze(0).detach().cpu().numpy().astype(np.float64))
            else:
                storage_path = storage_V0 + torch.cumsum(storage_delta, dim=1)
                projected_paths.append(storage_path.squeeze(0).detach().cpu().numpy().astype(np.float64))
            V_T = storage_path[:, -1, :]

            # Post-check: if pre-check says reachable but projection still misses
            # the requested hard target, keep hard target by default and report.
            # Optional fallback can be enabled in config for best-effort downgrade.
            req_residual_tensor = meta.get("requested_residual")
            if isinstance(req_residual_tensor, torch.Tensor):
                req_residual = req_residual_tensor.to(device=V_T.device, dtype=V_T.dtype)
                if req_residual.dim() == 1:
                    req_residual = req_residual.unsqueeze(0)
            else:
                req_residual = storage_VT - V_T
            post_policy = _update_post_projection_terminal_policy(
                req_residual=req_residual,
                V_terminal=V_T,
                initial_reachable_mask=initial_terminal_reachable_mask,
                current_reachable_mask=terminal_reachable_mask,
                current_best_effort_target=terminal_best_effort_target,
                reservoir_names=res_names,
                tol=terminal_reach_tol,
                allow_fallback=allow_post_projection_fallback,
            )
            terminal_reachable_mask = post_policy["reachable_mask"]
            terminal_best_effort_target = post_policy["best_effort_target"]
            for nm, gap_val in post_policy.get("hard_gap", {}).items():
                post_hard_gap[nm] = max(post_hard_gap.get(nm, 0.0), float(gap_val))
            for nm, gap_val in post_policy.get("fallback_gap", {}).items():
                post_unreachable[nm] = max(post_unreachable.get(nm, 0.0), float(gap_val))
            msg_parts = post_policy.get("message_parts", [])
            if msg_parts and not post_fallback_logged:
                if allow_post_projection_fallback:
                    print(
                        "[generate] year={} terminal target unreachable after projection coupling: {} "
                        "(switch to best-effort terminal attainment)".format(int(year), ", ".join(msg_parts))
                    )
                else:
                    print(
                        "[generate] year={} hard terminal target not met after projection: {} "
                        "(keep hard target; no fallback)".format(int(year), ", ".join(msg_parts))
                    )
                post_fallback_logged = True

            need = (terminal_best_effort_target - V_T).abs()
            K_use = max(0, min(window_k, q_phys.size(1)))
            if K_use > 0:
                cap = ((qmax[:, -K_use:, :] - qmin[:, -K_use:, :]) * dt_tensor[:, -K_use:, :] / 1e8).sum(dim=1)
            else:
                cap = torch.zeros_like(need)
            shortfall = torch.clamp(need - cap, min=0.0)
            infeasible = (shortfall > 0).float().mean().item()
            term_err = need.mean().item()
            ramp = ((q_phys[:, 1:] - q_phys[:, :-1]) ** 2).mean().item() if q_phys.size(1) > 1 else 0.0
            path_violation = ((storage_path < storage_Vmin) | (storage_path > storage_Vmax)).float().mean().item()

            meta_residual = meta.get("residual")
            if isinstance(meta_residual, torch.Tensor):
                residual_values.append(float(meta_residual.abs().max().item()))
            meta_infeasible = meta.get("infeasible")
            if isinstance(meta_infeasible, torch.Tensor):
                projection_infeasible.append(float(meta_infeasible.float().mean().item()))

            if violation > 0:
                raise AssertionError("inference produced out-of-bounds flow")
            if barrier_loss_fn is not None:
                barrier_val = barrier_loss_fn(q_phys, qmin, qmax).item()
                barrier_list.append(float(barrier_val))
            else:
                barrier_list.append(0.0)

            violation_list.append(float(violation))
            terminal_list.append(float(term_err))
            ramp_list.append(float(ramp))
            infeasible_list.append(float(infeasible))
            path_violation_list.append(float(path_violation))
            need_values.extend(need.detach().cpu().view(-1).tolist())
            cap_values.extend(cap.detach().cpu().view(-1).tolist())

    decoded_arr = np.asarray(decoded)
    if decoded_arr.size == 0:
        raise RuntimeError("No decoded schedules available for export.")
    mean_schedule = decoded_arr.mean(axis=0)
    try:
        distances = np.linalg.norm(decoded_arr - mean_schedule, axis=(1, 2))
        closest_idx = int(np.argmin(distances))
    except Exception:
        closest_idx = 0
    order_indices = [closest_idx] + [i for i in range(decoded_arr.shape[0]) if i != closest_idx]

    dt_np = dt_tensor.squeeze(0).detach().cpu().numpy().astype(np.float64).reshape(-1)
    storage_V0_np = storage_V0.squeeze(0).detach().cpu().numpy().astype(np.float64)
    head_np = head_inflow.squeeze(0).detach().cpu().numpy().astype(np.float64)
    if interval_inflow is not None:
        interval_np = interval_inflow.squeeze(0).detach().cpu().numpy().astype(np.float64)
    else:
        interval_np = np.zeros((T, max(0, R - 1)), dtype=np.float64)

    def schedule_df_export(
        schedule_np: np.ndarray,
        scheme_idx: int,
        storage_traj_np: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        scheme_name = f"scheme_{scheme_idx:02d}"
        period_labels = [f"{i + 1:02d}" for i in range(T)]
        schedule_np = schedule_np.astype(np.float64)
        if storage_traj_np is None:
            q_in_np_sched = np.zeros_like(schedule_np, dtype=np.float64)
            q_in_np_sched[:, 0] = head_np
            if interval_np.size > 0:
                for r in range(1, schedule_np.shape[1]):
                    q_in_np_sched[:, r] = schedule_np[:, r - 1] + interval_np[:, r - 1]
            else:
                for r in range(1, schedule_np.shape[1]):
                    q_in_np_sched[:, r] = schedule_np[:, r - 1]
            storage_delta_np = (q_in_np_sched - schedule_np) * dt_np[:, None] / 1e8
            storage_traj = storage_V0_np + np.cumsum(storage_delta_np, axis=0)
        else:
            storage_traj = np.asarray(storage_traj_np, dtype=np.float64)
            if storage_traj.shape != schedule_np.shape:
                raise ValueError("storage_traj_np shape must match schedule_np shape [T,R]")
        data: Dict[str, Any] = {
            "scheme": [scheme_name] * T,
            "period": period_labels,
        }
        flow_columns: List[str] = []
        storage_columns: List[str] = []
        for idx, name in enumerate(res_names):
            flow_col = f"{name}_出库流量(m3/s)"
            storage_col = f"{name}_库容(亿m³)"
            flow_vals = np.asarray(schedule_np[:, idx]).reshape(-1)
            vol_vals = np.asarray(storage_traj[:, idx]).reshape(-1)
            data[flow_col] = flow_vals
            data[storage_col] = vol_vals
            flow_columns.append(flow_col)
            storage_columns.append(storage_col)

        head_col = f"{res_names[0]}_入库流量(m3/s)"
        data[head_col] = np.asarray(head_np).reshape(-1)
        interval_columns: List[str] = []
        for i in range(R - 1):
            up = res_names[i]
            dn = res_names[i + 1]
            col_name = f"{up}-{dn}_区间来水(m3/s)"
            data[col_name] = np.asarray(interval_np[:, i]).reshape(-1)
            interval_columns.append(col_name)

        column_order = ["scheme", "period"] + flow_columns + storage_columns + [head_col] + interval_columns
        lengths = {k: len(np.asarray(v).reshape(-1)) for k, v in data.items()}
        ref_len = lengths.get("scheme", 0)
        mismatch = {k: l for k, l in lengths.items() if l != ref_len}
        if mismatch:
            raise ValueError(f"Column length mismatch: expected {ref_len}, got {mismatch}")
        return pd.DataFrame(data)[column_order]

    rows: List[pd.DataFrame] = []
    for export_idx, sample_idx in enumerate(order_indices[:sample_count]):
        st_np = projected_paths[sample_idx] if sample_idx < len(projected_paths) else None
        rows.append(schedule_df_export(decoded_arr[sample_idx], export_idx + 1, st_np))

    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = (_SCRIPT_DIR / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"diverse_schedules_{year}.csv"

    out_df = pd.concat(rows, axis=0, ignore_index=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[generate] year={year} saved UTF-8 file -> {out_path}")

    violation_rate = float(np.mean(np.asarray(violation_list) > 0)) if violation_list else 0.0
    terminal_error = float(np.mean(np.asarray(terminal_list))) if terminal_list else 0.0
    infeasible_ratio = float(np.mean(np.asarray(infeasible_list) > 0)) if infeasible_list else 0.0
    if allow_post_projection_fallback and post_unreachable:
        existing_unreachable = set(reach_report.get("unreachable_reservoirs", []) or [])
        per_res = reach_report.get("per_reservoir", {}) or {}
        for nm, gap_val in post_unreachable.items():
            existing_unreachable.add(nm)
            info = per_res.get(nm)
            if isinstance(info, dict):
                info["reachable"] = False
                info["status"] = "joint_infeasible_after_projection"
                info["gap_to_target"] = max(float(info.get("gap_to_target", 0.0)), float(gap_val))
        reach_report["unreachable_reservoirs"] = sorted(existing_unreachable)
        if per_res:
            total_res = max(1, len(per_res))
            reach_report["reachable_ratio"] = float((total_res - len(existing_unreachable)) / total_res)
            reach_report["hard_terminal_enforced_ratio"] = reach_report["reachable_ratio"]
        reach_report["max_gap"] = max(float(reach_report.get("max_gap", 0.0)), max(post_unreachable.values()))

    stats = _build_generation_stats(
        violation_rate=violation_rate,
        terminal_error=terminal_error,
        infeasible_ratio=infeasible_ratio,
        reach_report=reach_report,
        allow_post_projection_fallback=allow_post_projection_fallback,
        post_hard_gap=post_hard_gap,
    )

    stats_path = out_dir / f"diversity_stats_{year}.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return decoded, stats






