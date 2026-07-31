#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core helpers for level-informed projection during decoding."""

from __future__ import annotations

import torch

def slice_step(x: torch.Tensor, t: int) -> torch.Tensor:
    """Return step ``t`` of ``x`` supporting shapes [B,T,R] or [T,R]."""
    return x[:, t] if x.dim() == 3 else x[t]


def expand_br(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has batch dimension (convert [R] -> [1,R])."""
    return x.unsqueeze(0) if x.dim() == 1 else x


def q_box_from_level_step(
    curves,
    V_t: torch.Tensor,
    H_t: torch.Tensor,
    Qin_t: torch.Tensor,
    k_t: torch.Tensor,
    Hmin_t: torch.Tensor,
    Hmax_t: torch.Tensor,
    dH_up_t: torch.Tensor,
    dH_dn_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute flow bounds implied by level/ramp constraints for one step."""
    H_lo = torch.maximum(Hmin_t, H_t - dH_dn_t)
    H_hi = torch.minimum(Hmax_t, H_t + dH_up_t)
    V_lo = curves.h2v_all(H_lo)
    V_hi = curves.h2v_all(H_hi)
    k_t = k_t.clamp_min(1e-9)
    qL = Qin_t - (V_hi - V_t) / k_t
    qU = Qin_t - (V_lo - V_t) / k_t
    return torch.minimum(qL, qU), torch.maximum(qL, qU)


def reachability_shrink(
    qL: torch.Tensor,
    qU: torch.Tensor,
    V_t: torch.Tensor,
    Qin_future: torch.Tensor,
    qmin_future: torch.Tensor,
    qmax_future: torch.Tensor,
    k_future: torch.Tensor,
    V_T_lo: torch.Tensor,
    V_T_hi: torch.Tensor,
    alpha: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shrink flow bounds based on terminal reachability (optional)."""

    if Qin_future.numel() == 0 or Qin_future.size(-2) == 0:
        return qL, qU

    def _ensure_br(x: torch.Tensor) -> torch.Tensor:
        return x if x.dim() == 3 else x.unsqueeze(0)

    Qin_future = _ensure_br(Qin_future)
    qmin_future = _ensure_br(qmin_future)
    qmax_future = _ensure_br(qmax_future)
    k_future = _ensure_br(k_future)
    V_T_lo = V_T_lo if V_T_lo.dim() == 3 else V_T_lo.unsqueeze(0)
    V_T_hi = V_T_hi if V_T_hi.dim() == 3 else V_T_hi.unsqueeze(0)
    V_t = V_t if V_t.dim() == 3 else V_t.unsqueeze(1)

    k_future = k_future.clamp_min(1e-9)
    V_gain_if_qmin = ((Qin_future - qmin_future) * k_future).sum(dim=1)
    V_gain_if_qmax = ((Qin_future - qmax_future) * k_future).sum(dim=1)
    V_T_max = V_t.squeeze(1) + V_gain_if_qmin
    V_T_min = V_t.squeeze(1) + V_gain_if_qmax

    V_lo = torch.maximum(V_T_min, V_T_lo.squeeze(1))
    V_hi = torch.minimum(V_T_max, V_T_hi.squeeze(1))

    width_all = (V_T_max - V_T_min).clamp_min(1e-9)
    width_tar = (V_hi - V_lo).clamp_min(0.0)
    ratio = (width_tar / width_all).clamp(0.0, 1.0)

    mid = 0.5 * (qL + qU)
    half = 0.5 * (qU - qL) * (1.0 - alpha + alpha * ratio)
    return mid - half, mid + half


def soft_clip_projection(
    q_raw: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    q_box_lo: torch.Tensor,
    q_box_hi: torch.Tensor,
) -> torch.Tensor:
    """Intersection of existing flow box with level-induced box via softclip."""
    qL = torch.maximum(q_min, q_box_lo)
    qU = torch.minimum(q_max, q_box_hi)
    return softclip(q_raw, qL, qU)


def softclip(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, tau: float = 1e-2) -> torch.Tensor:
    """Smoothly project ``x`` into ``[lo, hi]`` preserving gradients."""
    lo, hi = torch.minimum(lo, hi), torch.maximum(lo, hi)
    span = (hi - lo).clamp_min(1e-12)
    z = (x - lo) / span
    z = torch.sigmoid((z - 0.5) / (tau + 1e-12))
    return lo + z * span


__all__ = [
    "slice_step",
    "expand_br",
    "q_box_from_level_step",
    "reachability_shrink",
    "soft_clip_projection",
]
