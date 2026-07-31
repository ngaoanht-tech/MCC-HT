#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helpers for deriving level and ramp limits used during decoding projection.
"""

from __future__ import annotations

import torch


def big_like(x: torch.Tensor, val: float = 1e9) -> torch.Tensor:
    """Return a tensor filled with ``val`` that matches ``x``'s shape/device/dtype."""
    return torch.full_like(x, float(val))


def compute_level_limits(
    *,
    V_min: torch.Tensor | None,
    V_max: torch.Tensor | None,
    H_min: torch.Tensor | None,
    H_max: torch.Tensor | None,
    dH_up_cfg: torch.Tensor | None,
    dH_dn_cfg: torch.Tensor | None,
    curves,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct per-step level bounds and ramp limits.

    Parameters
    ----------
    V_min, V_max : torch.Tensor or None
        Storage bounds [T,R] (亿 m³). Used as a fallback when explicit level
        bounds are unavailable.
    H_min, H_max : torch.Tensor or None
        Level bounds [T,R] (m).
    dH_up_cfg, dH_dn_cfg : torch.Tensor or None
        Optional per-step level change limits [T,R] (m). If missing, large
        defaults are used (i.e., no ramp restriction).
    curves : TorchCurves
        Provides ``v2h_all``/``h2v_all`` conversions.

    Returns
    -------
    (H_min_t, H_max_t, dH_up_t, dH_down_t) : tuple(torch.Tensor)
        All tensors shaped [T,R].
    """

    if H_min is not None and H_max is not None:
        Hmin_t = H_min
        Hmax_t = H_max
    elif V_min is not None and V_max is not None:
        Hmin_t = curves.v2h_all(V_min)
        Hmax_t = curves.v2h_all(V_max)
    else:
        raise ValueError("compute_level_limits requires either level or volume bounds.")

    if dH_up_cfg is None:
        dH_up_t = big_like(Hmin_t)
    else:
        dH_up_t = dH_up_cfg

    if dH_dn_cfg is None:
        dH_dn_t = big_like(Hmin_t)
    else:
        dH_dn_t = dH_dn_cfg

    return Hmin_t, Hmax_t, dH_up_t, dH_dn_t


def make_level_limits(
    *,
    curves,
    V_min: torch.Tensor | None,
    V_max: torch.Tensor | None,
    H_min: torch.Tensor | None,
    H_max: torch.Tensor | None,
    ramp_source: str = "none",
    ramp_up: torch.Tensor | None = None,
    ramp_dn: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convenience wrapper that falls back to unbounded ramps when absent."""
    Hmin_t, Hmax_t, _, _ = compute_level_limits(
        V_min=V_min,
        V_max=V_max,
        H_min=H_min,
        H_max=H_max,
        dH_up_cfg=None,
        dH_dn_cfg=None,
        curves=curves,
    )
    ramp_source = str(ramp_source or "none").lower()
    if ramp_source == "none" or ramp_up is None or ramp_dn is None:
        dH_up = big_like(Hmin_t)
        dH_dn = big_like(Hmin_t)
    else:
        dH_up = ramp_up
        dH_dn = ramp_dn
    return Hmin_t, Hmax_t, dH_up, dH_dn


__all__ = ["compute_level_limits", "make_level_limits", "big_like"]
