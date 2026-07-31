#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for analysis/visualization scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "diverse_results"
REAL_RESULTS_DIR = BASE_DIR / "real_results"
FIGURES_DIR = RESULTS_DIR / "figs"
CONFIG_PATH = BASE_DIR / "config.yaml"
CONSTRAINT_PATHS = (
    BASE_DIR / "shuxing" / "constraints_config.json",
    BASE_DIR / "constraint" / "constraints_config.json",
)

DEFAULT_RESERVOIRS = [
    "\u4e4c\u4e1c\u5fb7",  # 乌东德
    "\u767d\u9e64\u6ee9",  # 白鹤滩
    "\u6eaa\u6d1b\u6e21",  # 溪洛渡
    "\u5411\u5bb6\u575d",  # 向家坝
    "\u4e09\u5ce1",        # 三峡
    "\u845b\u6d32\u575d",  # 葛洲坝
]

_CONFIG_CACHE: Optional[dict] = None
_CONSTRAINT_CACHE: Optional[dict] = None


def _normalise_token(text: Union[str, float, int]) -> str:
    """Normalize text for fuzzy column matching."""
    s = str(text or "").strip().lower()
    replacements = {
        "\uFF08": "(",
        "\uFF09": ")",
        " ": "",
        "\u00b3": "3",
        "m^3": "m3",
        "m\u00b3": "m3",
        "m3/s": "m3/s",
        "\u2014": "-",
        "\u2013": "-",
        "_": "",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def load_config(refresh: bool = False) -> dict:
    """Load config.yaml once and cache the dictionary."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None or refresh:
        if not CONFIG_PATH.exists():
            _CONFIG_CACHE = {}
        else:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                _CONFIG_CACHE = yaml.safe_load(f) or {}
    return _CONFIG_CACHE.copy()


def load_constraints(refresh: bool = False) -> dict:
    """Load constraint configuration JSON from known locations."""
    global _CONSTRAINT_CACHE
    if _CONSTRAINT_CACHE is None or refresh:
        data = {}
        for path in CONSTRAINT_PATHS:
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    break
        _CONSTRAINT_CACHE = data
    return _CONSTRAINT_CACHE.copy()


def find_column(
    df: pd.DataFrame,
    keywords: Sequence[Union[str, Sequence[str]]],
    default: Optional[str] = None,
) -> str:
    """Find a column whose normalized name contains all keyword groups."""

    groups: List[List[str]] = []
    for token in keywords:
        if isinstance(token, (list, tuple, set)):
            groups.append([_normalise_token(x) for x in token])
        else:
            groups.append([_normalise_token(token)])

    for col in map(str, df.columns):
        col_norm = _normalise_token(col)
        ok = True
        for group in groups:
            if not any(g and g in col_norm for g in group):
                ok = False
                break
        if ok:
            return col

    if default is not None:
        return default
    joined = ", ".join("/".join(g) for g in groups)
    raise KeyError(f"column not found for keywords: {joined}")


def load_schedule_result(
    year: int,
    scheme: Optional[str] = None,
    prefer_xlsx: bool = True,
    sheet: str = "periods",
) -> pd.DataFrame:
    """Load generated schedules for a given year as DataFrame."""
    year = int(year)
    df: Optional[pd.DataFrame] = None

    if prefer_xlsx:
        xlsx = RESULTS_DIR / f"{year}\u5e74.xlsx"
        if xlsx.exists():
            df = pd.read_excel(xlsx, sheet_name=sheet)

    if df is None:
        csv_path = RESULTS_DIR / f"diverse_schedules_{year}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"generated schedule not found: {csv_path}")
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if "period" in df.columns:
        df["period"] = pd.to_numeric(df["period"], errors="coerce")
        df.sort_values("period", inplace=True)

    if scheme and "scheme" in df.columns:
        df = df[df["scheme"].astype(str) == str(scheme)].copy()
        if df.empty:
            raise ValueError(f"year {year} has no rows for scheme={scheme}")

    return df.reset_index(drop=True)


def load_real_data(year: int) -> pd.DataFrame:
    """Load historical observations for a year (real_results/{year}.csv)."""
    path = REAL_RESULTS_DIR / f"{int(year)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing real data: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    if "period" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "period"})

    df["period"] = pd.to_numeric(df["period"], errors="coerce")
    df.sort_values("period", inplace=True)
    return df.reset_index(drop=True)


def load_schedule_multi(year: int) -> pd.DataFrame:
    """Load all schemes for the given year."""
    return load_schedule_result(year, scheme=None)


__all__ = [
    "BASE_DIR",
    "RESULTS_DIR",
    "REAL_RESULTS_DIR",
    "FIGURES_DIR",
    "DEFAULT_RESERVOIRS",
    "load_config",
    "load_constraints",
    "find_column",
    "load_schedule_result",
    "load_schedule_multi",
    "load_real_data",
]
