#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d


_COL_CANDIDATES = {
    "level": ["水位", "水位m", "水位(M)", "Z", "H"],
    "volume": ["库容", "库容10^8", "库容(10^8", "V", "库容(亿m3)", "库容(亿m³)", "库容(亿)"],
}


class StrictWaterLevelConverter:
    """Strict H<->V converter from shuxing/curves without silent fallbacks."""

    def __init__(self, reservoir_order: List[str], curves_dir: str = "shuxing/curves", volume_scale: float = 1.0):
        self.reservoir_order = list(reservoir_order)
        self._h2v: Dict[str, interp1d] = {}
        self._v2h: Dict[str, interp1d] = {}
        self.device = torch.device("cpu")

        if not os.path.isdir(curves_dir):
            raise FileNotFoundError(f"未找到曲线目录: {curves_dir}")

        for name in self.reservoir_order:
            candidates = [
                os.path.join(curves_dir, f"{name}.csv"),
                os.path.join(curves_dir, f"{name}水位-库容曲线.csv"),
                os.path.join(curves_dir, f"{name}_水位-库容曲线.csv"),
            ]
            path = next((p for p in candidates if os.path.exists(p)), None)
            if path is None:
                raise FileNotFoundError(f"缺少水位-库容曲线文件: {candidates[0]}")
            df = pd.read_csv(path, encoding="utf-8-sig")

            def _pick_col(keys: List[str]) -> pd.Series:
                cols = [str(c).strip() for c in df.columns]
                for k in keys:
                    for c in cols:
                        if k in c:
                            return df[c]
                raise KeyError(f"{name} 曲线缺少列，期望含任一字段：{keys}；实际列={list(df.columns)}")

            h = _pick_col(_COL_CANDIDATES["level"]).astype(float).to_numpy()
            V = _pick_col(_COL_CANDIDATES["volume"]).astype(float).to_numpy() * float(volume_scale)

            order = np.argsort(h)
            h, V = h[order], V[order]
            eps = 1e-8
            for i in range(1, len(h)):
                if h[i] <= h[i - 1]:
                    h[i] = h[i - 1] + eps
            for i in range(1, len(V)):
                if V[i] <= V[i - 1]:
                    V[i] = V[i - 1] + eps

            self._h2v[name] = interp1d(h, V, kind="linear", fill_value="extrapolate", assume_sorted=True)
            self._v2h[name] = interp1d(V, h, kind="linear", fill_value="extrapolate", assume_sorted=True)

    def to(self, device: torch.device) -> "StrictWaterLevelConverter":
        self.device = device
        return self

    def h2s(self, levels: torch.Tensor, reservoir_idx: int) -> torch.Tensor:
        name = self.reservoir_order[reservoir_idx]
        v = self._h2v[name](levels.detach().cpu().numpy())
        return torch.as_tensor(v, dtype=levels.dtype, device=levels.device)

    def s2h(self, storage: torch.Tensor, reservoir_idx: int) -> torch.Tensor:
        name = self.reservoir_order[reservoir_idx]
        h = self._v2h[name](storage.detach().cpu().numpy())
        return torch.as_tensor(h, dtype=storage.dtype, device=storage.device)

    def to_volume(self, levels: torch.Tensor) -> torch.Tensor:
        x = levels
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B, R = x.shape
        outs = [self.h2s(x[:, r], r) for r in range(R)]
        return torch.stack(outs, dim=1)

    def to_level(self, volumes: torch.Tensor) -> torch.Tensor:
        v = volumes
        if v.dim() == 1:
            v = v.unsqueeze(0)
        B, R = v.shape
        outs = [self.s2h(v[:, r], r) for r in range(R)]
        return torch.stack(outs, dim=1)

