#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import torch


def _pick_col_for(df: pd.DataFrame, res_name: str, keyword_candidates: Sequence[str]) -> str:
    cols = list(map(str, df.columns))
    # 直接包含匹配
    for col in cols:
        if (res_name in col) and any((k in col) for k in keyword_candidates):
            return col
    # 常见乱码碎片（控制台/编码导致）
    garbles = ["涓嬮檺", "涓婇檺", "下限", "上限"]
    for col in cols:
        if (res_name in col) and any((g in col) for g in garbles):
            return col
    raise KeyError("缺少列: {} with {}".format(res_name, keyword_candidates))


def robust_load_storage_constraints(
    constraints_dir_or_path: Path,
    reservoir_names: Sequence[str],
    periods: int = 36,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Robustly load Vmin/Vmax columns with fuzzy matching and UTF-8-SIG.

    Returns: (Vmin[T,R], Vmax[T,R]) in 亿m³
    """
    path = Path(constraints_dir_or_path)
    if path.is_dir():
        # Prefer exact name; fallback to any CSV that has expected columns
        csv = path / "约束条件.csv"
        if not csv.exists():
            for cand in path.glob("*.csv"):
                try:
                    cols = pd.read_csv(str(cand), nrows=1, encoding="utf-8-sig").columns.tolist()
                    if any("库容下限" in c for c in cols) and any("库容上限" in c for c in cols):
                        csv = cand
                        break
                except Exception:
                    continue
        path = csv

    df = pd.read_csv(str(path), encoding="utf-8-sig")
    vmin_cols: List[str] = []
    vmax_cols: List[str] = []
    for nm in reservoir_names:
        vmin_cols.append(_pick_col_for(df, nm, ["库容下限", "下限"]))
        vmax_cols.append(_pick_col_for(df, nm, ["库容上限", "上限"]))

    Vmin_np = df[vmin_cols].iloc[:periods].to_numpy(dtype=float)
    Vmax_np = df[vmax_cols].iloc[:periods].to_numpy(dtype=float)
    Vmin_np = np.minimum(Vmin_np, Vmax_np)
    dev = device or torch.device("cpu")
    return (
        torch.tensor(Vmin_np, dtype=torch.float32, device=dev),
        torch.tensor(Vmax_np, dtype=torch.float32, device=dev),
    )

