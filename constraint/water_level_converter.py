"""Water level <-> storage conversion helpers.

This module keeps the legacy public API used by existing scripts while
providing a robust implementation that tolerates missing/partial configs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


def _build_interp_fn(x_raw: Iterable[float], y_raw: Iterable[float]):
    x = np.asarray(list(x_raw), dtype=np.float64).reshape(-1)
    y = np.asarray(list(y_raw), dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        raise ValueError("interpolation requires at least two finite points")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx]
    if x_unique.size < 2:
        raise ValueError("interpolation requires at least two unique x values")

    def _interp(values: np.ndarray) -> np.ndarray:
        return np.interp(values, x_unique, y_unique, left=y_unique[0], right=y_unique[-1])

    return _interp


def _as_level_tensor(arr_like: Sequence[Sequence[float]]) -> torch.Tensor:
    arr = np.asarray(arr_like, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("level constraints must be 2-D")
    # Expected internal shape is [R, T]. If [T, R], transpose.
    if arr.shape[0] == 36 and arr.shape[1] != 36:
        arr = arr.T
    return torch.tensor(arr, dtype=torch.float32)


class WaterLevelConverter:
    """Convert between level (m) and storage (1e8 m^3) by reservoir curves."""

    is_stub: bool = False

    def __init__(self, config_path: str = "constraint/constraints_config.json") -> None:
        self._module_dir = Path(__file__).resolve().parent
        self._project_dir = self._module_dir.parent
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = (self._project_dir / cfg_path).resolve()
        self.config_path = cfg_path
        self.config = self._load_config()

        self.vh_curves: Dict[str, object] = {}
        self.hv_curves: Dict[str, object] = {}
        self.reservoir_names: list[str] = []
        self._load_vh_curves()

        wl_cfg = self.config.get("water_level_constraints", {})
        hmin_raw = wl_cfg.get("Hmin")
        hmax_raw = wl_cfg.get("Hmax")
        if hmin_raw is not None and hmax_raw is not None:
            self.Hmin = _as_level_tensor(hmin_raw)
            self.Hmax = _as_level_tensor(hmax_raw)
        else:
            r = max(1, len(self.reservoir_names))
            self.Hmin = torch.zeros((r, 36), dtype=torch.float32)
            self.Hmax = torch.zeros((r, 36), dtype=torch.float32)

        self._ensure_identity_curves(max(1, int(self.Hmin.shape[0])))

        wb_cfg = self.config.get("water_balance_config", {})
        step_days = wb_cfg.get("time_step_days", [10.0] * 36)
        self.time_step_days = torch.tensor(step_days, dtype=torch.float32).reshape(-1)
        self.seconds_per_day = float(wb_cfg.get("seconds_per_day", 86400.0))
        self.time_step_seconds = self.time_step_days * self.seconds_per_day

        self.device = torch.device("cpu")

    def _ensure_identity_curves(self, reservoir_count: int) -> None:
        if self.reservoir_names:
            names = list(self.reservoir_names)
        else:
            names = [f"reservoir_{i}" for i in range(int(reservoir_count))]
            self.reservoir_names = names
        identity = _build_interp_fn([0.0, 1.0], [0.0, 1.0])
        for name in names:
            self.vh_curves.setdefault(name, identity)
            self.hv_curves.setdefault(name, identity)

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            return {}
        with self.config_path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)

    def _resolve_data_file(self, data_dir: str, filename: str) -> Optional[Path]:
        candidates = []
        f = Path(filename)
        if f.is_absolute():
            candidates.append(f)
        else:
            candidates.append((self._project_dir / data_dir / filename).resolve())
            candidates.append((self._module_dir / data_dir / filename).resolve())
            candidates.append((self._project_dir / filename).resolve())
        for path in candidates:
            if path.exists():
                return path
        return None

    def _read_curve_csv(self, path: Path) -> pd.DataFrame:
        for enc in ("utf-8-sig", "utf-8", "gbk"):
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue
        return pd.read_csv(path)

    def _load_vh_curves(self) -> None:
        conv_cfg = self.config.get("storage_level_conversion", {})
        data_dir = str(conv_cfg.get("data_source_dir", ""))
        vh_files = conv_cfg.get("vh_curves", {})
        if not isinstance(vh_files, dict):
            return

        for reservoir_name, filename in vh_files.items():
            if not isinstance(filename, str):
                continue
            path = self._resolve_data_file(data_dir, filename)
            if path is None:
                continue

            df = self._read_curve_csv(path)
            if df.shape[1] < 2:
                continue
            storage = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
            level = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
            try:
                self.vh_curves[reservoir_name] = _build_interp_fn(storage, level)
                self.hv_curves[reservoir_name] = _build_interp_fn(level, storage)
            except Exception:
                continue
            self.reservoir_names.append(str(reservoir_name))

    def to(self, device: torch.device | str):
        self.device = torch.device(device)
        self.Hmin = self.Hmin.to(self.device)
        self.Hmax = self.Hmax.to(self.device)
        self.time_step_days = self.time_step_days.to(self.device)
        self.time_step_seconds = self.time_step_seconds.to(self.device)
        return self

    def _curve_by_index(self, collection: Dict[str, object], reservoir_idx: int):
        if reservoir_idx < 0 or reservoir_idx >= len(self.reservoir_names):
            raise IndexError(f"reservoir_idx out of range: {reservoir_idx}")
        name = self.reservoir_names[reservoir_idx]
        fn = collection.get(name)
        if fn is None:
            raise ValueError(f"missing curve for reservoir: {name}")
        return fn

    def h2s(self, water_levels: torch.Tensor, reservoir_idx: int) -> torch.Tensor:
        fn = self._curve_by_index(self.hv_curves, reservoir_idx)
        shape = tuple(water_levels.shape)
        values = water_levels.detach().cpu().numpy().reshape(-1).astype(np.float64)
        storage = fn(values).reshape(shape)
        return torch.as_tensor(storage, dtype=water_levels.dtype, device=water_levels.device)

    def s2h(self, storage: torch.Tensor, reservoir_idx: int) -> torch.Tensor:
        fn = self._curve_by_index(self.vh_curves, reservoir_idx)
        shape = tuple(storage.shape)
        values = storage.detach().cpu().numpy().reshape(-1).astype(np.float64)
        levels = fn(values).reshape(shape)
        return torch.as_tensor(levels, dtype=storage.dtype, device=storage.device)

    # Aliases used by newer code paths.
    def to_volume(self, levels: torch.Tensor) -> torch.Tensor:
        if levels.ndim == 1:
            return torch.stack([self.h2s(levels[i : i + 1], i).squeeze(0) for i in range(levels.numel())], dim=0)
        if levels.ndim == 2:
            t, r = levels.shape
            out = torch.zeros_like(levels)
            for i in range(r):
                out[:, i] = self.h2s(levels[:, i], i)
            return out
        raise ValueError("levels must have shape [R] or [T,R]")

    def to_level(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim == 1:
            return torch.stack([self.s2h(volume[i : i + 1], i).squeeze(0) for i in range(volume.numel())], dim=0)
        if volume.ndim == 2:
            t, r = volume.shape
            out = torch.zeros_like(volume)
            for i in range(r):
                out[:, i] = self.s2h(volume[:, i], i)
            return out
        raise ValueError("volume must have shape [R] or [T,R]")

    def _dt_seconds(self, time_step_idx: int) -> torch.Tensor:
        idx = int(max(0, min(time_step_idx, int(self.time_step_seconds.numel()) - 1)))
        return self.time_step_seconds[idx]

    def compute_water_balance(
        self,
        initial_storage: torch.Tensor,
        inflow: torch.Tensor,
        outflow: torch.Tensor,
        time_step_idx: int,
    ) -> torch.Tensor:
        dt = self._dt_seconds(time_step_idx).to(initial_storage.device, dtype=initial_storage.dtype)
        return initial_storage + (inflow - outflow) * dt / 1e8

    def get_storage_constraints(self, time_step_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = int(max(0, min(time_step_idx, int(self.Hmin.shape[1]) - 1)))
        hmin_t = self.Hmin[:, t].to(self.device)
        hmax_t = self.Hmax[:, t].to(self.device)
        smin = torch.zeros_like(hmin_t)
        smax = torch.zeros_like(hmax_t)
        for i in range(len(self.reservoir_names)):
            smin[i] = self.h2s(hmin_t[i : i + 1], i).item()
            smax[i] = self.h2s(hmax_t[i : i + 1], i).item()
        return smin, smax

    def compute_storage_violation(self, storage: torch.Tensor, time_step_idx: int) -> torch.Tensor:
        smin, smax = self.get_storage_constraints(time_step_idx)
        smin = smin.unsqueeze(0).expand_as(storage)
        smax = smax.unsqueeze(0).expand_as(storage)
        return torch.relu(smin - storage) + torch.relu(storage - smax)

    def project_to_storage_constraints(self, storage: torch.Tensor, time_step_idx: int) -> torch.Tensor:
        smin, smax = self.get_storage_constraints(time_step_idx)
        smin = smin.unsqueeze(0).expand_as(storage)
        smax = smax.unsqueeze(0).expand_as(storage)
        return torch.clamp(storage, min=smin, max=smax)

    def get_initial_storage(self, batch_size: int = 1) -> torch.Tensor:
        levels = (
            self.config.get("water_balance_config", {})
            .get("initial_water_levels", {})
            .get("default_levels", [])
        )
        if levels:
            level_tensor = torch.tensor(levels, dtype=torch.float32)
        elif self.Hmax.numel() > 0:
            level_tensor = self.Hmax[:, 0].detach().cpu()
        else:
            level_tensor = torch.zeros((len(self.reservoir_names),), dtype=torch.float32)
        initial = self.to_volume(level_tensor)
        return initial.unsqueeze(0).expand(int(batch_size), -1)


class DataFrameWaterLevelConverter(WaterLevelConverter):
    """Build converter directly from in-memory curve DataFrames."""

    def __init__(
        self,
        static_curves: Dict[str, pd.DataFrame],
        reservoir_names: Optional[Sequence[str]] = None,
    ) -> None:
        self._module_dir = Path(__file__).resolve().parent
        self._project_dir = self._module_dir.parent
        self.config_path = self._project_dir / "constraint" / "constraints_config.json"
        self.config = {}
        self.vh_curves = {}
        self.hv_curves = {}
        self.reservoir_names = list(reservoir_names) if reservoir_names is not None else list(static_curves.keys())
        self.device = torch.device("cpu")
        self.time_step_days = torch.ones((36,), dtype=torch.float32)
        self.seconds_per_day = 86400.0
        self.time_step_seconds = self.time_step_days * self.seconds_per_day
        self.Hmin = torch.zeros((len(self.reservoir_names), 36), dtype=torch.float32)
        self.Hmax = torch.zeros((len(self.reservoir_names), 36), dtype=torch.float32)

        for name in self.reservoir_names:
            if name not in static_curves:
                continue
            df = static_curves[name]
            if df.shape[1] < 2:
                continue
            storage = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
            level = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
            try:
                self.vh_curves[name] = _build_interp_fn(storage, level)
                self.hv_curves[name] = _build_interp_fn(level, storage)
            except Exception:
                continue
