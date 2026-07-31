#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Callable, Dict, Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path


class WeightedMSELoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = (preds - targets) ** 2
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class RampLoss(nn.Module):
    def __init__(self, weight: float = 0.0) -> None:
        super().__init__()
        self.weight = float(weight)

    def forward(self, flows: torch.Tensor) -> torch.Tensor:
        if self.weight <= 0.0:
            return flows.new_tensor(0.0)
        diff = torch.diff(flows, dim=1)
        return self.weight * diff.abs().mean()


class InteriorBarrierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, q_phys: torch.Tensor, q_min: torch.Tensor, q_max: torch.Tensor) -> torch.Tensor:
        lo_gap = (q_phys - q_min).clamp_min(self.eps)
        hi_gap = (q_max - q_phys).clamp_min(self.eps)
        return (-torch.log(lo_gap) - torch.log(hi_gap)).mean()


class StoragePathConstraintLoss(nn.Module):
    def __init__(self, mode: str = "hinge", eps: float = 1e-8):
        super().__init__()
        assert mode in ("hinge", "barrier")
        self.mode = mode
        self.eps = float(eps)

    def forward(self, V_traj: torch.Tensor, Vmin: torch.Tensor, Vmax: torch.Tensor) -> torch.Tensor:
        if self.mode == "hinge":
            lo = (Vmin - V_traj).clamp_min(0.0)
            hi = (V_traj - Vmax).clamp_min(0.0)
            return (lo + hi).mean()
        lo_gap = (V_traj - Vmin).clamp_min(self.eps)
        hi_gap = (Vmax - V_traj).clamp_min(self.eps)
        return (-torch.log(lo_gap) - torch.log(hi_gap)).mean()


class TerminalWindowLoss(nn.Module):
    def __init__(self, window_k: int = 12, use_level: bool = False, eps: float = 1e-8):
        super().__init__()
        self.window_k = int(window_k)
        self.use_level = bool(use_level)
        self.eps = float(eps)

    def forward(
        self,
        q_phys: torch.Tensor,      # [B,T,R] outflow (m³/s)
        q_in: torch.Tensor,        # [B,T,R] inflow  (m³/s)
        V0: torch.Tensor,          # [B,R]   initial storage (亿m³)
        V_target: torch.Tensor,    # [B,R]   target storage  (亿m³)
        delta_t_steps: torch.Tensor,   # [T] seconds per period
        to_level: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        q_min: Optional[torch.Tensor] = None,
        q_max: Optional[torch.Tensor] = None,
        window_k: Optional[int] = None,
    ) -> torch.Tensor:
        B, T, R = q_phys.shape
        k = min(int(window_k or self.window_k), T)
        if delta_t_steps is None or delta_t_steps.numel() != T:
            raise ValueError("delta_t_steps 必须提供并长度等?T")
        dt = delta_t_steps.to(q_phys.device, dtype=q_phys.dtype).view(1, T, 1)
        q_in_win = q_in[:, -k:, :]
        q_out_win = q_phys[:, -k:, :]
        dt_win = dt[:, -k:, :]
        deltaV = torch.sum((q_in_win - q_out_win) * dt_win, dim=1) / 1e8  # 亿m³
        VT = V0 + deltaV
        if self.use_level and (to_level is not None):
            hT = to_level(VT)
            hTgt = to_level(V_target)
            err = hT - hTgt
        else:
            err = VT - V_target
        if (q_min is not None) and (q_max is not None):
            span = (q_max[:, -k:, :] - q_min[:, -k:, :]).clamp_min(self.eps)
            cap = torch.sum(span * dt_win, dim=1) / 1e8
            scale = cap.median(dim=1, keepdim=True).values.clamp_min(self.eps)
            err = err / scale
        return torch.mean(err ** 2)


# [DEPRECATED] Legacy placeholder retained for backward compatibility.
class PowerOptimizedLoss(nn.Module):
    def __init__(self, power_weight: float = 0.0, flow_weight: float = 0.0):
        super().__init__()
        self.power_weight = float(power_weight)
        self.flow_weight = float(flow_weight)

    def _power_generation(self, flows: torch.Tensor) -> torch.Tensor:
        return flows.sum(dim=-1)

    def forward(self, flows: torch.Tensor) -> torch.Tensor:
        if self.power_weight <= 0.0:
            return flows.new_tensor(0.0)
        return -self.power_weight * self._power_generation(flows).mean()


class TerminalToGoLoss(nn.Module):
    """Rolling terminal gap penalty across the tail window (per-reservoir aware)."""

    def __init__(self, window_k: int = 36, weight_profile: str = "linear") -> None:
        super().__init__()
        self.window_k = int(window_k)
        self.weight_profile = str(weight_profile).lower()

    def _tail_weights(self, K: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.weight_profile == "quadratic":
            base = torch.linspace(0.0, 1.0, steps=K, device=device, dtype=dtype)
            return (base.pow(2) * 0.5 + 0.5).view(1, K, 1)
        if self.weight_profile == "cubic":
            base = torch.linspace(0.0, 1.0, steps=K, device=device, dtype=dtype)
            return (base.pow(3) * 0.5 + 0.5).view(1, K, 1)
        return torch.linspace(0.5, 1.0, steps=K, device=device, dtype=dtype).view(1, K, 1)

    def forward(
        self,
        V: torch.Tensor,
        V_target: torch.Tensor,
        reservoir_weights: Optional[torch.Tensor] = None,
        reservoir_window: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if V.dim() != 3:
            raise ValueError("V must have shape [B, T, R]")
        B, T, R = V.shape
        if reservoir_window is None:
            window_vec = torch.full((R,), min(self.window_k, T), device=V.device, dtype=torch.long)
        else:
            window_vec = reservoir_window.to(device=V.device, dtype=torch.long).clamp(1, T)
        losses = []
        V_target = V_target.to(V.dtype)
        for r in range(R):
            K = int(window_vec[r].item())
            idx0 = T - K
            V_tail = V[:, idx0:, r]
            target_r = V_target[:, r].view(B, 1)
            gap = (target_r - V_tail).clamp_min(0.0)
            tail_weights = self._tail_weights(K, V.device, V.dtype)
            loss_r = (gap.pow(2) * tail_weights.squeeze(-1)).mean()
            if reservoir_weights is not None:
                loss_r = loss_r * reservoir_weights[r]
            losses.append(loss_r)
        if not losses:
            return V.new_tensor(0.0)
        return torch.stack(losses).mean()


class LateStageMarginLoss(nn.Module):
    """Encourage headroom above Vmin in the terminal window (per-reservoir aware)."""

    def __init__(self, window_k: int = 36, margin: float = 0.05, weight_profile: str = "linear") -> None:
        super().__init__()
        self.window_k = int(window_k)
        self.margin = float(margin)
        self.weight_profile = str(weight_profile).lower()

    def _tail_weights(self, K: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.weight_profile == "quadratic":
            base = torch.linspace(0.0, 1.0, steps=K, device=device, dtype=dtype)
            return (base.pow(2) * 0.5 + 0.5).view(1, K, 1)
        if self.weight_profile == "cubic":
            base = torch.linspace(0.0, 1.0, steps=K, device=device, dtype=dtype)
            return (base.pow(3) * 0.5 + 0.5).view(1, K, 1)
        return torch.linspace(0.5, 1.0, steps=K, device=device, dtype=dtype).view(1, K, 1)

    def forward(
        self,
        V: torch.Tensor,
        Vmin: torch.Tensor,
        per_reservoir_margin: Optional[torch.Tensor] = None,
        reservoir_weights: Optional[torch.Tensor] = None,
        reservoir_window: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if V.dim() != 3:
            raise ValueError("V must have shape [B, T, R]")
        B, T, R = V.shape
        if Vmin.dim() == 2:
            Vmin = Vmin.unsqueeze(0).expand(B, -1, -1)
        elif Vmin.dim() == 3 and Vmin.size(0) == 1 and B > 1:
            Vmin = Vmin.expand(B, -1, -1)
        elif Vmin.dim() != 3:
            raise ValueError("Vmin must have shape [T,R] or [B,T,R]")
        Vmin = Vmin.to(device=V.device, dtype=V.dtype)
        if reservoir_window is None:
            window_vec = torch.full((R,), min(self.window_k, T), device=V.device, dtype=torch.long)
        else:
            window_vec = reservoir_window.to(device=V.device, dtype=torch.long).clamp(1, T)
        losses = []
        for r in range(R):
            K = int(window_vec[r].item())
            idx0 = T - K
            V_tail = V[:, idx0:, r]
            Vmin_tail = Vmin[:, idx0:, r]
            margin_r = self.margin
            if per_reservoir_margin is not None:
                margin_r = float(per_reservoir_margin[r].item())
            margin_gap = (Vmin_tail + margin_r - V_tail).clamp_min(0.0)
            tail_weights = self._tail_weights(K, V.device, V.dtype)
            loss_r = (margin_gap.pow(2) * tail_weights.squeeze(-1)).mean()
            if reservoir_weights is not None:
                loss_r = loss_r * reservoir_weights[r]
            losses.append(loss_r)
        if not losses:
            return V.new_tensor(0.0)
        return torch.stack(losses).mean()


class TorchLUT1D(nn.Module):
    """Simple differentiable 1-D table lookup with linear interpolation."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("TorchLUT1D expects 1-D inputs")
        if x.size < 2:
            raise ValueError("TorchLUT1D requires at least two support points")
        order = np.argsort(x)
        x_sorted = x[order].copy()
        y_sorted = y[order]
        # Ensure strictly increasing abscissa for stable interpolation
        for i in range(1, x_sorted.size):
            if x_sorted[i] <= x_sorted[i - 1]:
                x_sorted[i] = np.nextafter(x_sorted[i - 1], np.float32(np.inf))
        self.register_buffer("xk", torch.from_numpy(x_sorted))
        self.register_buffer("yk", torch.from_numpy(y_sorted))

    def forward(self, xq: torch.Tensor) -> torch.Tensor:
        if xq.numel() == 0:
            return torch.empty_like(xq)
        device = self.xk.device
        dtype = self.xk.dtype
        xq_flat = xq.to(device=device, dtype=dtype).reshape(-1).contiguous()
        idx = torch.searchsorted(self.xk, xq_flat)
        idx = torch.clamp(idx, 1, self.xk.numel() - 1)
        x0 = self.xk[idx - 1]
        x1 = self.xk[idx]
        y0 = self.yk[idx - 1]
        y1 = self.yk[idx]
        denom = (x1 - x0).clamp_min(torch.finfo(dtype).eps)
        w = (xq_flat - x0) / denom
        yq = y0 + w * (y1 - y0)
        return yq.reshape_as(xq.to(device=device, dtype=dtype))


class TorchCurves(nn.Module):
    """Load reservoir V-H and Q-Zt curves as differentiable LUTs."""

    def __init__(self, curves_dir: str, tailwater_dir: str, reservoir_names: Sequence[str]) -> None:
        super().__init__()
        curves_path = Path(curves_dir)
        tail_path = Path(tailwater_dir)
        if not curves_path.is_dir():
            raise FileNotFoundError(f"curves_dir not found: {curves_dir}")
        if not tail_path.is_dir():
            raise FileNotFoundError(f"tailwater_dir not found: {tailwater_dir}")
        self.names = list(reservoir_names)
        v2h_modules = []
        h2v_modules = []
        q2zt_modules = []
        for name in self.names:
            vh_candidates = [
                curves_path / f"{name}.csv",
                curves_path / f"{name}水位-库容曲线.csv",
                curves_path / f"{name}_水位-库容曲线.csv",
            ]
            vh_file = next((p for p in vh_candidates if p.exists()), None)
            if vh_file is None:
                raise FileNotFoundError(f"Missing H-V curve for {name} in {curves_path}")
            df_vh = pd.read_csv(vh_file, encoding="utf-8-sig")
            if df_vh.shape[1] < 2:
                raise ValueError(f"H-V curve file {vh_file} must contain at least two columns")
            vol = df_vh.iloc[:, 0].astype(np.float32).to_numpy()
            lvl = df_vh.iloc[:, 1].astype(np.float32).to_numpy()
            mask = np.isfinite(vol) & np.isfinite(lvl)
            if not np.any(mask):
                raise ValueError(f"H-V curve {vh_file} contains no finite values")
            v2h_modules.append(TorchLUT1D(vol[mask], lvl[mask]))
            h2v_modules.append(TorchLUT1D(lvl[mask], vol[mask]))

            tw_file = tail_path / f"{name}出库流量-下游水位曲线.csv"
            if not tw_file.exists():
                raise FileNotFoundError(f"Missing Q-Zt curve for {name}: {tw_file}")
            df_tw = pd.read_csv(tw_file, encoding="utf-8-sig")
            if df_tw.shape[1] < 2:
                raise ValueError(f"Q-Zt curve file {tw_file} must contain at least two columns")
            flow = df_tw.iloc[:, 0].astype(np.float32).to_numpy()
            tail = df_tw.iloc[:, 1].astype(np.float32).to_numpy()
            mask_tw = np.isfinite(flow) & np.isfinite(tail)
            if not np.any(mask_tw):
                raise ValueError(f"Q-Zt curve {tw_file} contains no finite values")
            q2zt_modules.append(TorchLUT1D(flow[mask_tw], tail[mask_tw]))

        self.v2h = nn.ModuleList(v2h_modules)
        self.h2v = nn.ModuleList(h2v_modules)
        self.q2zt = nn.ModuleList(q2zt_modules)

    def v2h_all(self, storage: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.v2h[idx](storage[..., idx]) for idx in range(storage.size(-1))], dim=-1)

    def h2v_all(self, level: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.h2v[idx](level[..., idx]) for idx in range(level.size(-1))], dim=-1)

    def q2zt_all(self, flow: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.q2zt[idx](flow[..., idx]) for idx in range(flow.size(-1))], dim=-1)


class PowerEnergySurrogate(nn.Module):
    """Compute average turbine output (GW) using differentiable curves."""

    def __init__(self, curves: TorchCurves, rho: float = 1000.0, g: float = 9.81, eta: float = 0.88) -> None:
        super().__init__()
        self.curves = curves
        self.rho = float(rho)
        self.g = float(g)
        self.eta = float(eta)
        self._per_reservoir_scale: Optional[torch.Tensor] = None

    def set_per_reservoir_scale(self, scale: torch.Tensor) -> None:
        self._per_reservoir_scale = scale.detach().clone()

    def forward(
        self,
        q_out: torch.Tensor,
        q_in: torch.Tensor,
        V0: torch.Tensor,
        dt_steps: torch.Tensor,
    ) -> torch.Tensor:
        if q_out.dim() != 3:
            raise ValueError("q_out must have shape [B, T, R]")
        dt = dt_steps.view(1, -1, 1).to(q_out)
        dt_hours = dt / 3600.0
        volume = V0.unsqueeze(1) + torch.cumsum((q_in - q_out) * dt / 1e8, dim=1)
        head = self.curves.v2h_all(volume)
        tail = self.curves.q2zt_all(q_out)
        net_head = (head - tail).clamp_min(0.0)
        power_watts = self.rho * self.g * self.eta * q_out * net_head
        if self._per_reservoir_scale is not None:
            scale = self._per_reservoir_scale.to(device=q_out.device, dtype=q_out.dtype)
            power_watts = power_watts * scale.view(1, 1, -1)
        power_gw = power_watts / 1e9
        weighted = power_gw * dt_hours
        denom = dt_hours.sum(dim=(1, 2)).clamp_min(1e-6)
        avg_power = weighted.sum(dim=(1, 2)) / denom
        return avg_power


class CompositeSchedulingLoss(nn.Module):
    def __init__(
        self,
        *,
        flow_loss: Optional[nn.Module] = None,
        terminal_loss: Optional[TerminalWindowLoss] = None,
        ramp_loss: Optional[RampLoss] = None,
        barrier_loss: Optional[InteriorBarrierLoss] = None,
        power_surrogate: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        w_flow: float = 1.0,
        w_term: float = 0.0,
        w_ramp: float = 0.0,
        w_power: float = 0.0,
        w_barrier: float = 0.0,
        w_need: float = 0.0,
        w_path: float = 0.0,
        path_mode: str = "hinge",
        w_term_to_go: float = 0.0,
        w_late_margin: float = 0.0,
        late_margin_value: float = 0.0,
        water_level_curves: Optional["TorchCurves"] = None,
        w_level_path: float = 0.0,
        w_level_ramp: float = 0.0,
        level_path_reservoir_weights: Optional[Sequence[float]] = None,
        level_ramp_reservoir_weights: Optional[Sequence[float]] = None,
        terminal_mode: str = "volume_window",
        terminal_level_target: Optional[Sequence[float]] = None,
        terminal_level_tolerance: float = 0.0,
        terminal_level_reservoir_weights: Optional[Sequence[float]] = None,
        reserve_enabled: bool = False,
        w_reserve: float = 0.0,
        reserve_k_tail: int = 0,
        reserve_margin: float = 0.0,
        reserve_reservoir_weights: Optional[Sequence[float]] = None,
        term_to_go_reservoir_weights: Optional[Sequence[float]] = None,
        term_to_go_window_override: Optional[Sequence[int]] = None,
        late_margin_reservoir_weights: Optional[Sequence[float]] = None,
        late_margin_values: Optional[Sequence[float]] = None,
        late_margin_window_override: Optional[Sequence[int]] = None,
        tail_weight_profile: str = "linear",
        need_reservoir_weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.flow_loss = flow_loss or WeightedMSELoss()
        self.terminal_loss = terminal_loss
        self.ramp_loss = ramp_loss
        self.barrier_loss = barrier_loss
        self.power_surrogate = power_surrogate
        self.w_flow = float(w_flow)
        self.w_term = float(w_term)
        self.w_ramp = float(w_ramp)
        self.w_power = float(w_power)
        self.w_barrier = float(w_barrier)
        self.w_need = float(w_need)
        self.w_path = float(w_path)
        self._path_loss = StoragePathConstraintLoss(mode=path_mode) if self.w_path > 0 else None
        self.w_term_to_go = float(w_term_to_go)
        self.w_late_margin = float(w_late_margin)
        self.water_level_curves = water_level_curves
        self.w_level_path = float(w_level_path)
        self.w_level_ramp = float(w_level_ramp)
        self._level_path_loss = StoragePathConstraintLoss(mode=path_mode) if self.w_level_path > 0 else None
        self.level_path_reservoir_weights = list(level_path_reservoir_weights or [])
        self.level_ramp_reservoir_weights = list(level_ramp_reservoir_weights or [])
        self.terminal_mode = str(terminal_mode or "volume_window")
        self.terminal_level_target = list(terminal_level_target or [])
        self.terminal_level_tolerance = float(terminal_level_tolerance)
        self.terminal_level_reservoir_weights = list(terminal_level_reservoir_weights or [])
        self.reserve_enabled = bool(reserve_enabled)
        self.w_reserve = float(w_reserve)
        self.reserve_k_tail = int(reserve_k_tail)
        self.reserve_margin = float(reserve_margin)
        self.reserve_reservoir_weights = list(reserve_reservoir_weights or [])
        self.term_window_k = int(getattr(self.terminal_loss, "window_k", 36))
        self.tail_weight_profile = str(tail_weight_profile)
        self.term_to_go_reservoir_weights = list(term_to_go_reservoir_weights or [])
        self.term_to_go_window_override = list(term_to_go_window_override or [])
        self.late_margin_reservoir_weights = list(late_margin_reservoir_weights or [])
        self.late_margin_values_override = list(late_margin_values or [])
        self.late_margin_window_override = list(late_margin_window_override or [])
        self.default_late_margin_value = float(late_margin_value)
        self.term_to_go_loss = (
            TerminalToGoLoss(window_k=self.term_window_k, weight_profile=self.tail_weight_profile)
            if self.w_term_to_go > 0
            else None
        )
        self.late_stage_margin_loss = (
            LateStageMarginLoss(
                window_k=self.term_window_k,
                margin=self.default_late_margin_value,
                weight_profile=self.tail_weight_profile,
            )
            if self.w_late_margin > 0
            else None
        )
        self.need_reservoir_weights = list(need_reservoir_weights or [])

    @staticmethod
    def _resolve_reservoir_tensor(
        raw: Any,
        R: int,
        device: torch.device,
        dtype: torch.dtype,
        default_value: float,
    ) -> torch.Tensor:
        if raw is None:
            values_raw: Sequence[Any] = ()
        elif isinstance(raw, (list, tuple)):
            values_raw = raw
        else:
            values_raw = (raw,)
        if not values_raw:
            return torch.full((R,), float(default_value), device=device, dtype=dtype)
        values = torch.tensor([float(x) for x in values_raw], device=device, dtype=dtype)
        if values.numel() < R:
            pad_count = R - values.numel()
            pad_value = float(values[-1].item()) if values.numel() > 0 else float(default_value)
            pad = torch.full((pad_count,), pad_value, device=device, dtype=dtype)
            values = torch.cat([values, pad], dim=0)
        elif values.numel() > R:
            values = values[:R]
        return values

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        *,
        physical_predictions: Optional[torch.Tensor] = None,
        q_in: Optional[torch.Tensor] = None,
        V0: Optional[torch.Tensor] = None,
        V_target: Optional[torch.Tensor] = None,
        q_min: Optional[torch.Tensor] = None,
        q_max: Optional[torch.Tensor] = None,
        window_k: Optional[int] = None,
        to_level: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        delta_t_steps: Optional[torch.Tensor] = None,
        V_min: Optional[torch.Tensor] = None,
        V_max: Optional[torch.Tensor] = None,
        terminal_effective_target: Optional[torch.Tensor] = None,
        return_term_per_sample: bool = False,
    ) -> Any:
        q_phys = physical_predictions if physical_predictions is not None else preds
        total = preds.new_tensor(0.0)
        term_per_sample = None
        V_target_loss = terminal_effective_target if terminal_effective_target is not None else V_target
        if self.w_flow:
            total = total + self.w_flow * self.flow_loss(preds, targets)
        if (
            self.terminal_loss is not None
            and self.w_term
            and self.terminal_mode != "level_window"
            and all(x is not None for x in (q_in, V0, V_target_loss, delta_t_steps))
        ):
            total = total + self.w_term * self.terminal_loss(
                q_phys=q_phys,
                q_in=q_in,
                V0=V0,
                V_target=V_target_loss,
                delta_t_steps=delta_t_steps,
                to_level=to_level,
                q_min=q_min,
                q_max=q_max,
                window_k=window_k,
            )
        if self.ramp_loss is not None and self.w_ramp:
            total = total + self.w_ramp * self.ramp_loss(q_phys)
        if self.barrier_loss is not None and self.w_barrier and (q_min is not None) and (q_max is not None):
            total = total + self.w_barrier * self.barrier_loss(q_phys, q_min, q_max)
        if self.w_power and self.power_surrogate is not None:
            total = total + (-self.w_power) * self.power_surrogate(q_phys)
        need_v_traj = (
            (self._path_loss is not None and self.w_path and all(x is not None for x in (q_in, V0, delta_t_steps, V_min, V_max)))
            or (self.term_to_go_loss is not None and self.w_term_to_go and all(x is not None for x in (q_in, V0, V_target_loss, delta_t_steps)))
            or (self.late_stage_margin_loss is not None and self.w_late_margin and all(x is not None for x in (q_in, V0, V_min, delta_t_steps)))
            or (
                self.water_level_curves is not None
                and (self.w_level_path > 0 or self.w_level_ramp > 0 or (self.reserve_enabled and self.w_reserve > 0))
                and all(x is not None for x in (q_in, V0, delta_t_steps))
            )
            or (self.reserve_enabled and self.w_reserve > 0 and all(x is not None for x in (q_in, V0, V_target_loss, delta_t_steps)))
        )
        V_traj = None
        if need_v_traj:
            dt = delta_t_steps.to(q_phys.device, dtype=q_phys.dtype).view(1, -1, 1)
            delta_V = (q_in - q_phys) * dt / 1e8
            V_traj = V0.unsqueeze(1) + torch.cumsum(delta_V, dim=1)

        if (
            V_traj is not None
            and self.w_term
            and self.terminal_mode == "level_window"
            and self.water_level_curves is not None
        ):
            R = V_traj.size(-1)
            device = V_traj.device
            dtype = V_traj.dtype
            if self.terminal_level_target:
                target_levels = torch.tensor(self.terminal_level_target, device=device, dtype=dtype)
            else:
                target_levels = torch.zeros((R,), device=device, dtype=dtype)
            tol = float(self.terminal_level_tolerance)
            h_lo = target_levels - tol
            h_hi = target_levels + tol
            V_lo = self.water_level_curves.h2v_all(h_lo)
            V_hi = self.water_level_curves.h2v_all(h_hi)
            V_T = V_traj[:, -1, :]
            low_gap = (V_lo.view(1, -1) - V_T).clamp_min(0.0)
            high_gap = (V_T - V_hi.view(1, -1)).clamp_min(0.0)
            term_penalty = low_gap.pow(2) + high_gap.pow(2)
            if self.terminal_level_reservoir_weights:
                weight_vec = self._resolve_reservoir_tensor(
                    self.terminal_level_reservoir_weights, R, device, dtype, 1.0
                )
                term_penalty = term_penalty * weight_vec.view(1, -1)
            total = total + self.w_term * term_penalty.mean()

        V_target_exp = None
        if V_traj is not None and V_target_loss is not None:
            if V_target_loss.dim() == 1:
                V_target_exp = V_target_loss.unsqueeze(0).expand(V_traj.size(0), -1)
            elif V_target_loss.dim() == 2 and V_target_loss.size(0) == 1 and V_traj.size(0) > 1:
                V_target_exp = V_target_loss.expand(V_traj.size(0), -1)
            else:
                V_target_exp = V_target_loss

        # 单库末期最大误差惩罚：防止误差集中到某一个水库
        if V_traj is not None and V_target_exp is not None and getattr(self, "w_term_max", 0.0) > 0.0:
            V_T_last = V_traj[:, -1, :]  # [B,R]
            gap = V_T_last - V_target_exp  # [B,R]
            gap_sq = gap.pow(2)
            per_sample_max = gap_sq.max(dim=1).values  # [B]
            max_term = per_sample_max.mean()
            total = total + float(self.w_term_max) * max_term

        if V_traj is not None and V_target_exp is not None and self.w_need > 0.0:
            need_gap = (V_target_exp - V_traj[:, -1, :]).abs()
            if self.need_reservoir_weights:
                weight_vec = self._resolve_reservoir_tensor(
                    self.need_reservoir_weights,
                    need_gap.size(-1),
                    need_gap.device,
                    need_gap.dtype,
                    1.0,
                )
                need_gap = need_gap * weight_vec.view(1, -1)
            total = total + self.w_need * need_gap.mean()

        if V_traj is not None and self._path_loss is not None and self.w_path and all(x is not None for x in (V_min, V_max)):
            if V_min.dim() == 2:
                V_min_exp = V_min.unsqueeze(0)
            elif V_min.dim() == 3:
                V_min_exp = V_min
            else:
                raise ValueError("V_min must have shape [T,R] or [B,T,R]")
            if V_max.dim() == 2:
                V_max_exp = V_max.unsqueeze(0)
            elif V_max.dim() == 3:
                V_max_exp = V_max
            else:
                raise ValueError("V_max must have shape [T,R] or [B,T,R]")
            V_min_exp = V_min_exp.to(V_traj.dtype)
            V_max_exp = V_max_exp.to(V_traj.dtype)
            if V_min_exp.size(0) == 1 and V_traj.size(0) > 1:
                V_min_exp = V_min_exp.expand(V_traj.size(0), -1, -1)
            if V_max_exp.size(0) == 1 and V_traj.size(0) > 1:
                V_max_exp = V_max_exp.expand(V_traj.size(0), -1, -1)
            total = total + self.w_path * self._path_loss(V_traj, V_min_exp, V_max_exp)

        if V_traj is not None and self.term_to_go_loss is not None and self.w_term_to_go and V_target_exp is not None:
            R = V_traj.size(-1)
            device = V_traj.device
            dtype = V_traj.dtype
            weight_vec = None
            if self.term_to_go_reservoir_weights:
                weight_vec = self._resolve_reservoir_tensor(
                    self.term_to_go_reservoir_weights, R, device, dtype, 1.0
                )
            window_vec = None
            if self.term_to_go_window_override:
                window_vec = self._resolve_reservoir_tensor(
                    self.term_to_go_window_override, R, device, dtype, self.term_window_k
                ).long()
            total = total + self.w_term_to_go * self.term_to_go_loss(V_traj, V_target_exp, weight_vec, window_vec)

        if V_traj is not None and self.late_stage_margin_loss is not None and self.w_late_margin and V_min is not None:
            if V_min.dim() == 2:
                Vmin_for_margin = V_min.unsqueeze(0)
            elif V_min.dim() == 3:
                Vmin_for_margin = V_min
            else:
                raise ValueError("V_min must have shape [T,R] or [B,T,R]")
            if Vmin_for_margin.size(0) == 1 and V_traj.size(0) > 1:
                Vmin_for_margin = Vmin_for_margin.expand(V_traj.size(0), -1, -1)
            R = V_traj.size(-1)
            device = V_traj.device
            dtype = V_traj.dtype
            margin_vec = None
            if self.late_margin_values_override:
                margin_vec = self._resolve_reservoir_tensor(
                    self.late_margin_values_override, R, device, dtype, self.default_late_margin_value
                )
            weight_vec = None
            if self.late_margin_reservoir_weights:
                weight_vec = self._resolve_reservoir_tensor(
                    self.late_margin_reservoir_weights, R, device, dtype, 1.0
                )
            window_vec = None
            if self.late_margin_window_override:
                window_vec = self._resolve_reservoir_tensor(
                    self.late_margin_window_override, R, device, dtype, self.term_window_k
                ).long()
            total = total + self.w_late_margin * self.late_stage_margin_loss(
                V_traj,
                Vmin_for_margin,
                margin_vec,
                weight_vec,
                window_vec,
            )
        if (
            V_traj is not None
            and V_target_exp is not None
            and self.reserve_enabled
            and self.w_reserve > 0
            and self.reserve_k_tail > 1
        ):
            tail_len = min(self.reserve_k_tail, V_traj.size(1))
            if tail_len > 1:
                desired = (V_target_exp - self.reserve_margin).clamp_min(0.0)
                tail_slice = V_traj[:, -tail_len:-1, :]
                deficit = (desired.unsqueeze(1) - tail_slice).clamp_min(0.0)
                R = V_traj.size(-1)
                device = V_traj.device
                dtype = V_traj.dtype
                if self.reserve_reservoir_weights:
                    reserve_vec = self._resolve_reservoir_tensor(
                        self.reserve_reservoir_weights, R, device, dtype, 1.0
                    )
                    deficit = deficit * reserve_vec.view(1, 1, -1)
                total = total + self.w_reserve * deficit.pow(2).mean()
        if (
            V_traj is not None
            and self.water_level_curves is not None
            and (self.w_level_path > 0 or self.w_level_ramp > 0)
        ):
            H_traj = self.water_level_curves.v2h_all(V_traj)
            R = H_traj.size(-1)
            device = H_traj.device
            dtype = H_traj.dtype
            level_path_vec = None
            if self.level_path_reservoir_weights:
                level_path_vec = self._resolve_reservoir_tensor(
                    self.level_path_reservoir_weights, R, device, dtype, 1.0
                )
            level_ramp_vec = None
            if self.level_ramp_reservoir_weights:
                level_ramp_vec = self._resolve_reservoir_tensor(
                    self.level_ramp_reservoir_weights, R, device, dtype, 1.0
                )

            if self.w_level_path > 0 and V_min is not None and V_max is not None:
                if V_min.dim() == 2:
                    Vmin_level = V_min.unsqueeze(0)
                elif V_min.dim() == 3:
                    Vmin_level = V_min
                else:
                    raise ValueError("V_min must have shape [T,R] or [B,T,R]")
                if Vmin_level.size(0) == 1 and V_traj.size(0) > 1:
                    Vmin_level = Vmin_level.expand(V_traj.size(0), -1, -1)
                Vmin_level = Vmin_level.to(V_traj.dtype)

                if V_max.dim() == 2:
                    Vmax_level = V_max.unsqueeze(0)
                elif V_max.dim() == 3:
                    Vmax_level = V_max
                else:
                    raise ValueError("V_max must have shape [T,R] or [B,T,R]")
                if Vmax_level.size(0) == 1 and V_traj.size(0) > 1:
                    Vmax_level = Vmax_level.expand(V_traj.size(0), -1, -1)
                Vmax_level = Vmax_level.to(V_traj.dtype)

                H_min = self.water_level_curves.v2h_all(Vmin_level)
                H_max = self.water_level_curves.v2h_all(Vmax_level)
                violation_low = (H_min - H_traj).clamp_min(0.0)
                violation_high = (H_traj - H_max).clamp_min(0.0)
                level_penalty = violation_low.pow(2) + violation_high.pow(2)
                if level_path_vec is not None:
                    level_penalty = level_penalty * level_path_vec.view(1, 1, -1)
                total = total + self.w_level_path * level_penalty.mean()

            if self.w_level_ramp > 0 and H_traj.size(1) > 1:
                delta_h = (H_traj[:, 1:, :] - H_traj[:, :-1, :]).abs()
                if level_ramp_vec is not None:
                    delta_h = delta_h * level_ramp_vec.view(1, 1, -1)
                total = total + self.w_level_ramp * delta_h.mean()
        if return_term_per_sample and term_per_sample is not None:
            return total, term_per_sample
        return total


def calculate_performance_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, Any]:
    with torch.no_grad():
        err = (y_pred - y_true).float()
        mse = err.pow(2).mean()
        rmse = torch.sqrt(mse)
        mae = err.abs().mean()

        y_true_flat = y_true.float().reshape(-1)
        y_pred_flat = y_pred.float().reshape(-1)
        if y_true_flat.numel() > 1:
            y_true_mean = y_true_flat.mean()
            ss_tot = (y_true_flat - y_true_mean).pow(2).sum()
            ss_res = (y_true_flat - y_pred_flat).pow(2).sum()
            ss_tot_val = ss_tot.item()
            if ss_tot_val > 0:
                r2_value = 1.0 - ss_res.item() / ss_tot_val
            else:
                r2_value = 0.0

            if y_true_flat.std(unbiased=False) > 0 and y_pred_flat.std(unbiased=False) > 0:
                corr_matrix = torch.corrcoef(torch.stack([y_true_flat, y_pred_flat]))
                corr_value = corr_matrix[0, 1].item()
            else:
                corr_value = 0.0
        else:
            r2_value = 0.0
            corr_value = 0.0

        mape = ((y_pred_flat - y_true_flat).abs() / torch.clamp(y_true_flat.abs(), min=1e-6)).mean()

        return {
            "mse": mse.item(),
            "rmse": rmse.item(),
            "mae": mae.item(),
            "r2": r2_value,
            "corr": corr_value,
            "mape": mape.item(),
        }
