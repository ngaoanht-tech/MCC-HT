#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature engineering and lightweight preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import warnings

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

try:
    from config_loader import get_config
except Exception:  # pragma: no cover - optional dependency
    get_config = None  # type: ignore


DEFAULT_RESERVOIRS = (
    "乌东德",
    "白鹤滩",
    "溪洛渡",
    "向家坝",
    "三峡",
    "葛洲坝",
)

_TERMINAL_GUIDANCE_WARNED = False


def _safe_get_config() -> Dict[str, Any]:
    if get_config is None:
        return {}
    try:
        cfg = get_config()
        return cfg if isinstance(cfg, dict) else getattr(cfg, "config", {})
    except Exception:
        return {}


@dataclass
class TransformationState:
    inflow_scaler: Optional[StandardScaler] = None
    outflow_scaler: Optional[StandardScaler] = None
    feature_scaler: Optional[StandardScaler] = None


class FeatureEngineer:
    """Builds input features for the multiscale transformer."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or _safe_get_config()
        preprocess_cfg = cfg.get("preprocessing", {}) if isinstance(cfg, dict) else {}

        self.use_log_transform = bool(preprocess_cfg.get("use_log_transform", True))
        self.normalize = bool(preprocess_cfg.get("normalize", True))
        reservoirs = cfg.get("reservoirs", {}) if isinstance(cfg, dict) else {}
        self.reservoir_names = tuple(reservoirs.get("names", DEFAULT_RESERVOIRS))

    # ------------------------------------------------------------------
    def create_features(
        self,
        inflow_data: np.ndarray,
        outflow_data: Optional[np.ndarray] = None,
        df: Optional[Any] = None,
    ) -> np.ndarray:
        """Create a 22-dimensional causal feature vector per record."""

        inflow_data = np.asarray(inflow_data, dtype=np.float32)
        n_samples = inflow_data.shape[0]

        features = [inflow_data]  # 6 cols

        inflow_mean = inflow_data.mean(axis=1, keepdims=True)
        inflow_std = inflow_data.std(axis=1, keepdims=True)
        features.extend([inflow_mean, inflow_std])  # 2 cols

        time_features = self._time_features(n_samples)
        features.append(time_features)  # 4 cols

        lag_features = self._lag_features(inflow_data)
        features.append(lag_features)  # 2 cols

        corr_features = self._correlation_features(inflow_data)
        features.append(corr_features)  # 3 cols

        cascade_features = self._cascade_features(inflow_data)
        features.append(cascade_features)  # 3 cols

        ratio_features = self._ratio_features(inflow_data)
        features.append(ratio_features)  # 2 cols

        matrix = np.hstack(features).astype(np.float32)
        return matrix

    # ------------------------------------------------------------------
    def _time_features(self, n_samples: int) -> np.ndarray:
        out = np.zeros((n_samples, 4), dtype=np.float32)
        for i in range(n_samples):
            period = (i % 36) + 1
            out[i, 0] = np.sin(2 * np.pi * period / 36.0)
            out[i, 1] = np.cos(2 * np.pi * period / 36.0)
            out[i, 2] = period / 36.0
            out[i, 3] = 1.0 if period <= 18 else 0.0
        return out

    def _lag_features(self, inflow: np.ndarray) -> np.ndarray:
        n_samples = inflow.shape[0]
        out = np.zeros((n_samples, 2), dtype=np.float32)
        if n_samples > 1:
            out[1:, 0] = inflow[:-1, 0]
        if n_samples > 2:
            out[2:, 1] = inflow[:-2, 0]
        return out

    def _correlation_features(self, inflow: np.ndarray) -> np.ndarray:
        n_samples = inflow.shape[0]
        out = np.zeros((n_samples, 3), dtype=np.float32)
        for i in range(4, n_samples):
            upstream = inflow[i - 4 : i + 1, 0]
            lead = upstream[:-1]
            lag = upstream[1:]
            if np.std(lead) > 1e-6 and np.std(lag) > 1e-6:
                out[i, 0] = np.corrcoef(lead, lag)[0, 1]
        if n_samples > 1:
            main_diff = np.diff(inflow[:, 0])
            if inflow.shape[1] > 1:
                tributary_diff = np.diff(inflow[:, 1:].sum(axis=1))
                mask = (np.abs(main_diff) > 0.1) & (np.abs(tributary_diff) > 0.1)
                out[1:, 1] = np.sign(main_diff * tributary_diff) * mask
        for i in range(n_samples):
            flows = inflow[i]
            total = flows.sum()
            if total > 1e-6:
                weights = flows / total
                sorted_w = np.sort(weights)
                n = len(sorted_w)
                index = np.arange(1, n + 1)
                out[i, 2] = (2 * np.sum(index * sorted_w)) / (n * np.sum(sorted_w)) - (n + 1) / n
        return out

    def _cascade_features(self, inflow: np.ndarray) -> np.ndarray:
        n_samples = inflow.shape[0]
        out = np.zeros((n_samples, 3), dtype=np.float32)
        cumulative = np.cumsum(inflow[:, 0])
        out[:, 0] = cumulative / (np.arange(n_samples) + 1)
        if n_samples > 2:
            for i in range(2, n_samples):
                upstream_avg = inflow[i - 2 : i, 0].mean()
                downstream_now = inflow[i, 1:].mean() if inflow.shape[1] > 1 else inflow[i, 0]
                out[i, 1] = downstream_now / upstream_avg if upstream_avg > 1e-6 else 0.0
        for i in range(n_samples):
            flows = inflow[i]
            mean_flow = flows.mean()
            out[i, 2] = np.std(flows) / mean_flow if mean_flow > 1e-6 else 0.0
        return out

    def _ratio_features(self, inflow: np.ndarray) -> np.ndarray:
        n_samples = inflow.shape[0]
        out = np.zeros((n_samples, 2), dtype=np.float32)
        for i in range(n_samples):
            main_flow = inflow[i, 0]
            tributary_flow = inflow[i, 1:].sum() if inflow.shape[1] > 1 else 0.0
            out[i, 0] = main_flow / tributary_flow if tributary_flow > 1e-6 else 0.0
            total_in = inflow[i].sum()
            out[i, 1] = main_flow / total_in if total_in > 1e-6 else 0.0
        return out
class DataTransformer:
    """Lightweight log/standardisation transformer."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or _safe_get_config()
        preprocess_cfg = cfg.get("preprocessing", {}) if isinstance(cfg, dict) else {}
        self.use_log_transform = bool(preprocess_cfg.get("use_log_transform", True))
        self.normalize = bool(preprocess_cfg.get("normalize", True))

        self.inflow_scaler: Optional[StandardScaler] = None
        self.outflow_scaler: Optional[StandardScaler] = None
        self.feature_scaler: Optional[StandardScaler] = None

    def fit_transform(
        self,
        inflow: np.ndarray,
        outflow: np.ndarray,
        features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        inflow_t = np.asarray(inflow, dtype=np.float32)
        outflow_t = np.asarray(outflow, dtype=np.float32)
        feature_t = np.asarray(features, dtype=np.float32)

        if self.use_log_transform:
            inflow_t = np.log1p(np.maximum(inflow_t, 0.0))
            outflow_t = np.log1p(np.maximum(outflow_t, 0.0))

        if self.normalize:
            self.inflow_scaler = StandardScaler().fit(inflow_t)
            self.outflow_scaler = StandardScaler().fit(outflow_t)
            self.feature_scaler = StandardScaler().fit(feature_t)
            inflow_t = self.inflow_scaler.transform(inflow_t)
            outflow_t = self.outflow_scaler.transform(outflow_t)
            feature_t = self.feature_scaler.transform(feature_t)

        return inflow_t.astype(np.float32), outflow_t.astype(np.float32), feature_t.astype(np.float32)

    def transform(
        self,
        inflow: np.ndarray,
        outflow: np.ndarray,
        features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        inflow_t = np.asarray(inflow, dtype=np.float32)
        outflow_t = np.asarray(outflow, dtype=np.float32)
        feature_t = np.asarray(features, dtype=np.float32)

        if self.use_log_transform:
            inflow_t = np.log1p(np.maximum(inflow_t, 0.0))
            outflow_t = np.log1p(np.maximum(outflow_t, 0.0))

        if self.normalize:
            if self.inflow_scaler is not None:
                inflow_t = self.inflow_scaler.transform(inflow_t)
            if self.outflow_scaler is not None:
                outflow_t = self.outflow_scaler.transform(outflow_t)
            if self.feature_scaler is not None:
                feature_t = self.feature_scaler.transform(feature_t)

        return inflow_t.astype(np.float32), outflow_t.astype(np.float32), feature_t.astype(np.float32)

    def inverse_transform_outflow(self, outflow_normalized: np.ndarray) -> np.ndarray:
        data = np.asarray(outflow_normalized, dtype=np.float32)
        if self.normalize and self.outflow_scaler is not None:
            data = self.outflow_scaler.inverse_transform(data)
        if self.use_log_transform:
            data = np.expm1(data)
        return data

    def export_transforms(self) -> Dict[str, object]:
        return {
            "inflow_scaler": self.inflow_scaler,
            "outflow_scaler": self.outflow_scaler,
            "feature_scaler": self.feature_scaler,
            "use_log_transform": self.use_log_transform,
            "normalize": self.normalize,
        }

    def save_transforms(self, filepath: str) -> None:
        joblib.dump(self.export_transforms(), filepath)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "DataTransformer":
        inst = cls()
        inst.inflow_scaler = data.get("inflow_scaler")  # type: ignore[assignment]
        inst.outflow_scaler = data.get("outflow_scaler")  # type: ignore[assignment]
        inst.feature_scaler = data.get("feature_scaler")  # type: ignore[assignment]
        inst.use_log_transform = bool(data.get("use_log_transform", True))
        inst.normalize = bool(data.get("normalize", True))
        return inst

    def load_state(self, data: Dict[str, object]) -> None:
        inst = self.__class__.from_dict(data)
        self.inflow_scaler = inst.inflow_scaler
        self.outflow_scaler = inst.outflow_scaler
        self.feature_scaler = inst.feature_scaler
        self.use_log_transform = inst.use_log_transform
        self.normalize = inst.normalize

    # Backwards-compatibility aliases ---------------------------------
    @classmethod
    def load_transforms(cls, data: Dict[str, object]) -> "DataTransformer":
        return cls.from_dict(data)

    def load_transforms_inplace(self, data: Dict[str, object]) -> None:
        self.load_state(data)

    def check_consistency(self) -> bool:
        return not self.normalize or self.feature_scaler is not None



def build_multiscale_features(
    inputs: Dict[str, torch.Tensor],
    cfg: Optional[Dict[str, Any]] = None,
) -> Optional[torch.Tensor]:
    """
    Construct multiscale statistics for cascaded reservoirs.

    Returns
    -------
    torch.Tensor or None
        Tensor shaped [B, T, R, C] where C = (#metrics * #windows [+ extras]).
        Returns ``None`` if required tensors are missing.
    """

    q_in = inputs.get("q_in")
    if q_in is None:
        return None

    if q_in.dim() == 2:
        q_in = q_in.unsqueeze(0)
    if q_in.dim() != 3:
        raise ValueError("q_in must have shape [B,T,R] or [T,R]")

    q_in = q_in.to(dtype=torch.float32)
    B, T, R = q_in.shape
    device = q_in.device

    def _cfg_get(path: str, default: Any) -> Any:
        if cfg is None:
            return default
        current: Any = cfg
        parts = path.split(".")
        for idx, part in enumerate(parts):
            last = idx == len(parts) - 1
            if hasattr(current, "get"):
                current = current.get(part, default if last else {})
            elif isinstance(current, dict):
                current = current.get(part, default if last else {})
            else:
                current = getattr(current, part, default if last else {})
            if current is None:
                return default if last else {}
        return current

    windows_cfg = _cfg_get("multiscale.windows", [3, 6, 12, 36])
    windows: List[int] = sorted({int(w) for w in (windows_cfg or []) if int(w) > 0})
    if not windows:
        windows = [3, 6, 12, 36]

    reduces_cfg = _cfg_get("multiscale.reduces", ["mean", "std", "sum"])
    reduces = {str(r).lower() for r in (reduces_cfg or [])}
    # Ensure core statistics are always available
    reduces.update({"mean", "std"})

    def _prepare_optional(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        if t.dim() == 2:
            t = t.unsqueeze(0)
        if t.dim() != 3:
            raise ValueError("Expected optional tensors to have shape [B,T,R] or [T,R]")
        if t.size(0) == 1 and B > 1:
            t = t.expand(B, -1, -1)
        return t.to(device=device, dtype=q_in.dtype)

    q_min = _prepare_optional(inputs.get("q_min"))
    q_max = _prepare_optional(inputs.get("q_max"))
    # Optional terminal-guidance context
    delta_t = inputs.get("delta_t")  # [T] or [1,T] or [B,T]
    V0_opt = inputs.get("V0")       # [B,R] or [1,R]
    VT_opt = inputs.get("V_target")  # [B,R] or [1,R]
    Vmin_opt = _prepare_optional(inputs.get("V_min"))
    Vmax_opt = _prepare_optional(inputs.get("V_max"))

    eps = 1e-6
    features: List[torch.Tensor] = []

    for window in windows:
        window = min(window, T)
        means: List[torch.Tensor] = []
        stds: List[torch.Tensor] = []
        mins: List[torch.Tensor] = []
        maxs: List[torch.Tensor] = []
        sums: List[torch.Tensor] = []
        for t in range(T):
            start = max(0, t - window + 1)
            segment = q_in[:, start : t + 1, :]
            means.append(segment.mean(dim=1))
            stds.append(segment.std(dim=1, unbiased=False))
            mins.append(segment.min(dim=1).values)
            maxs.append(segment.max(dim=1).values)
            sums.append(segment.sum(dim=1))

        mean_tensor = torch.stack(means, dim=1)
        std_tensor = torch.stack(stds, dim=1)
        min_tensor = torch.stack(mins, dim=1)
        max_tensor = torch.stack(maxs, dim=1)
        sum_tensor = torch.stack(sums, dim=1)

        range_tensor = (max_tensor - min_tensor).clamp_min(eps)
        delta_tensor = q_in - mean_tensor
        norm_pos_tensor = (q_in - min_tensor) / range_tensor
        norm_pos_tensor = norm_pos_tensor.clamp(0.0, 1.0)

        if "mean" in reduces:
            features.append(mean_tensor.unsqueeze(-1))
        if "std" in reduces:
            features.append(std_tensor.unsqueeze(-1))
        if "min" in reduces:
            features.append(min_tensor.unsqueeze(-1))
        if "max" in reduces:
            features.append(max_tensor.unsqueeze(-1))
        if "sum" in reduces:
            features.append(sum_tensor.unsqueeze(-1))

        features.append(delta_tensor.unsqueeze(-1))
        features.append(norm_pos_tensor.unsqueeze(-1))

    if q_min is not None and q_max is not None:
        span = (q_max - q_min).clamp_min(eps)
        fill = (q_in - q_min) / span
        headroom = (q_max - q_in) / span
        features.append(fill.clamp(0.0, 1.0).unsqueeze(-1))
        features.append(headroom.clamp(0.0, 1.0).unsqueeze(-1))

    # Terminal-guidance features (remain_need_ratio, upper/lower slack),
    # computed using only constraints and inflows (no model outputs).
    try:
        if (
            (delta_t is not None)
            and (q_min is not None)
            and (V0_opt is not None)
            and (VT_opt is not None)
            and (Vmin_opt is not None)
            and (Vmax_opt is not None)
        ):
            if not torch.is_tensor(delta_t):
                delta_t = torch.tensor(delta_t, dtype=q_in.dtype, device=device)
            delta_t = delta_t.to(device=device, dtype=q_in.dtype)
            if delta_t.dim() == 1:
                dt = delta_t.view(1, T, 1).expand(B, -1, R)
            elif delta_t.dim() == 2:
                dt = delta_t.unsqueeze(-1).expand(B, -1, R)
            else:
                dt = delta_t.view(B, T, 1).expand(B, -1, R)
            dt_vol = dt / 1e8

            V0 = V0_opt.to(device=device, dtype=q_in.dtype)
            if V0.dim() == 2 and V0.size(0) == 1 and B > 1:
                V0 = V0.expand(B, -1)
            VT = VT_opt.to(device=device, dtype=q_in.dtype)
            if VT.dim() == 2 and VT.size(0) == 1 and B > 1:
                VT = VT.expand(B, -1)

            guidance_need_scale = _cfg_get("multiscale.guidance.need_ratio_scale", None)
            guidance_slack_scale = _cfg_get("multiscale.guidance.slack_scale", None)
            guidance_remain_scale = _cfg_get("multiscale.guidance.remain_ratio_scale", None)

            def _resolve_scale(raw_scale: Any) -> torch.Tensor:
                if raw_scale is None:
                    return torch.ones(R, dtype=q_in.dtype, device=device)
                if isinstance(raw_scale, (list, tuple)):
                    values = torch.tensor([float(x) for x in raw_scale], dtype=q_in.dtype, device=device)
                else:
                    values = torch.full((R,), float(raw_scale), dtype=q_in.dtype, device=device)
                if values.numel() < R:
                    values = torch.nn.functional.pad(values, (0, R - values.numel()), "replicate")
                elif values.numel() > R:
                    values = values[:R]
                return values

            need_scale_vec = _resolve_scale(guidance_need_scale).view(1, 1, R)
            slack_scale_vec = _resolve_scale(guidance_slack_scale).view(1, 1, R)
            remain_scale_vec = _resolve_scale(guidance_remain_scale).view(1, 1, R)

            # Reconstruct storage trajectory using predicted outflow (teacher-free guidance)
            q_out_pred = inputs.get("q_out_pred")
            q_out_tensor = _prepare_optional(q_out_pred) if q_out_pred is not None else None
            if q_out_tensor is None:
                q_out_tensor = q_min if q_min is not None else q_in

            storage_delta = (q_in - q_out_tensor) * dt_vol
            V_base = V0.unsqueeze(1) + torch.cumsum(storage_delta, dim=1)
            V_base = torch.maximum(Vmin_opt, torch.minimum(V_base, Vmax_opt))

            cap_rem = torch.flip((q_max - q_min) * dt_vol, dims=[1]).cumsum(dim=1)
            cap_rem = torch.flip(cap_rem, dims=[1])

            need_rem = (VT.unsqueeze(1) - V_base)
            need_ratio = (need_rem / (cap_rem + eps)).clamp(-2.0, 2.0) * need_scale_vec
            need_ratio = (need_ratio / 2.0).clamp(-1.0, 1.0)

            target_span = (VT - V0).clamp_min(eps)
            remain_ratio = ((VT.unsqueeze(1) - V_base).clamp_min(0.0) / (target_span.unsqueeze(1) + eps)).clamp(0.0, 1.5)
            remain_ratio = remain_ratio * remain_scale_vec
            remain_ratio = ((remain_ratio - 0.75) / 0.75).clamp(-1.0, 1.0)
            remain_ratio_signed = ((VT.unsqueeze(1) - V_base) / (target_span.unsqueeze(1) + eps)).clamp(-2.0, 2.0)
            remain_ratio_signed = (remain_ratio_signed / 2.0).clamp(-1.0, 1.0)

            spanV = (Vmax_opt - Vmin_opt).clamp_min(eps)
            slack_up = ((Vmax_opt - V_base) / spanV).clamp(0.0, 1.0) * slack_scale_vec
            slack_lo = ((V_base - Vmin_opt) / spanV).clamp(0.0, 1.0) * slack_scale_vec
            slack_up = ((slack_up - 0.5) / 0.5).clamp(-1.0, 1.0)
            slack_lo = ((slack_lo - 0.5) / 0.5).clamp(-1.0, 1.0)

            downstream_need_stack = []
            # Accumulate downstream need per time step -> shape [B, T].
            running_need = torch.zeros_like(need_rem[:, :, 0])
            for r_idx in range(R - 1, -1, -1):
                running_need = running_need + need_rem[:, :, r_idx]
                downstream_need_stack.append(running_need.clone())
            downstream_need = torch.stack(downstream_need_stack[::-1], dim=-1)
            downstream_need_norm = (downstream_need / (cap_rem + eps)).clamp(-2.0, 2.0)
            downstream_need_norm = (downstream_need_norm / 2.0).clamp(-1.0, 1.0)

            if T > 1:
                residual_frac = torch.linspace(0, 1, steps=T, device=device, dtype=q_in.dtype)
                residual_frac = residual_frac.flip(0).view(1, T, 1).expand(B, -1, R)
            else:
                residual_frac = torch.zeros((B, T, R), dtype=q_in.dtype, device=device)
            residual_frac = ((residual_frac - 0.5) / 0.5).clamp(-1.0, 1.0)

            if q_min is not None and q_max is not None:
                span_flow = (q_max - q_min).clamp_min(eps)
                net_flow_norm = ((q_out_tensor - q_min) / span_flow).clamp(-3.0, 3.0)
                net_flow_norm = (net_flow_norm / 3.0).clamp(-1.0, 1.0)
            else:
                net_flow_norm = torch.zeros_like(q_out_tensor)

            features.append(need_ratio.unsqueeze(-1))
            features.append(slack_up.unsqueeze(-1))
            features.append(slack_lo.unsqueeze(-1))
            features.append(remain_ratio.unsqueeze(-1))
            features.append(remain_ratio_signed.unsqueeze(-1))
            features.append(downstream_need_norm.unsqueeze(-1))
            features.append(residual_frac.unsqueeze(-1))
            features.append(net_flow_norm.unsqueeze(-1))
    except Exception as exc:
        # Terminal-guidance features are optional, but surface the first failure.
        global _TERMINAL_GUIDANCE_WARNED
        if not _TERMINAL_GUIDANCE_WARNED:
            warnings.warn(
                f"terminal guidance feature generation failed; falling back without it: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            _TERMINAL_GUIDANCE_WARNED = True

    if not features:
        return None

    return torch.cat(features, dim=-1)






