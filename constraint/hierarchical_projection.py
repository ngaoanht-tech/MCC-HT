#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Differentiable hierarchical projection for multi-reservoir scheduling.

The projection enforces:
    1. Flow box constraints:      q_min ≤ q_out ≤ q_max
    2. Storage prefix boxes:      V_min ≤ V_prefix ≤ V_max
    3. Terminal equality:         V(T-1) = V_target
    4. Mass balance relationships: V(t) = V0 + Σ (q_in - q_out) Δt

The implementation operates directly on torch tensors so gradients propagate
through to the upstream transformer parameters. The algorithm performs a
single pass clamp followed by a small number of global re-distributions so it
remains inexpensive while keeping numerical stability.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class HierarchicalProjection(nn.Module):
    """Project model outputs onto the joint flow/storage feasible set.

    Fix: ensure terminal equality is evaluated and re-balanced after the final
    flow clamp as well, so returned (q_proj, V_proj) reflect the terminal target
    within tolerance instead of only the pre-clamp cumulative path.
    """

    def __init__(
        self,
        max_iters: int = 2,
        eps: float = 1e-6,
        volume_scale: float = 1e8,
        post_clamp_rebalance_iters: int = 2,
        post_clamp_tol: float = 5e-4,
        terminal_second_rebalance_iters: int = 0,
        terminal_second_rebalance_window: int = 36,
        tail_focus_reservoirs: Optional[Sequence[int]] = None,
        tail_focus_time_decay: float = 0.0,
        tail_focus_extra_window: int = 0,
        tail_focus_step_fraction: float = 1.0,
    ):
        super().__init__()
        self.max_iters = int(max_iters)
        self.eps = float(eps)
        # The hydrologic configuration stores storage in 10^8 m^3 units.
        # Keep the scale explicit so we can convert cumulative sums properly.
        self.volume_scale = float(volume_scale)
        # Extra refinement rounds after the final flow clamp
        self.post_clamp_rebalance_iters = int(max(0, post_clamp_rebalance_iters))
        self.post_clamp_tol = float(post_clamp_tol)
        self.terminal_second_rebalance_iters = int(max(0, terminal_second_rebalance_iters))
        self.terminal_second_rebalance_window = int(max(1, terminal_second_rebalance_window))
        if tail_focus_reservoirs:
            focus_set = sorted({int(max(0, idx)) for idx in tail_focus_reservoirs})
            self.tail_focus_indices = tuple(focus_set)
        else:
            self.tail_focus_indices = tuple()
        self.tail_focus_time_decay = float(max(0.0, tail_focus_time_decay))
        self.tail_focus_extra_window = int(max(0, tail_focus_extra_window))
        step_fraction = float(tail_focus_step_fraction)
        if step_fraction <= 0.0:
            step_fraction = 1.0
        self.tail_focus_step_fraction = float(min(step_fraction, 1.0))

    @staticmethod
    def _expand_to_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Broadcast helper that safely expands constraint tensors to batch size."""
        if tensor.dim() == 2:  # [T, R]
            tensor = tensor.unsqueeze(0)
        if tensor.size(0) == 1 and tensor.size(0) != batch_size:
            tensor = tensor.expand(batch_size, -1, -1)
        return tensor

    @staticmethod
    def _expand_vector(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        if tensor.dim() == 1:
            tensor = tensor.view(1, -1, 1)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(-1)
        if tensor.size(0) == 1 and tensor.size(0) != batch_size:
            tensor = tensor.expand(batch_size, -1, -1)
        return tensor

    def forward(
        self,
        q_raw: torch.Tensor,
        q_in: torch.Tensor,
        V0: torch.Tensor,
        V_target: torch.Tensor,
        q_min: torch.Tensor,
        q_max: torch.Tensor,
        Vmin: torch.Tensor,
        Vmax: torch.Tensor,
        delta_t: torch.Tensor,
        terminal_reachable_mask: Optional[torch.Tensor] = None,
        terminal_best_effort_target: Optional[torch.Tensor] = None,
        curves: Optional = None,  # TorchCurves for V-H conversion
        level_limits: Optional[Dict[str, torch.Tensor]] = None,  # {H_min, H_max, dH_up, dH_dn}
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        q_raw : torch.Tensor
            Initial outflow predictions [B, T, R].
        q_in : torch.Tensor
            Inflow sequences [B, T, R].
        V0 : torch.Tensor
            Initial storages [B, R].
        V_target : torch.Tensor
            Target terminal storages [B, R].
        q_min, q_max : torch.Tensor
            Flow bounds. Accepts shapes [T, R] or [B, T, R].
        Vmin, Vmax : torch.Tensor
            Storage bounds per prefix. Accepts shapes [T, R] or [B, T, R].
        delta_t : torch.Tensor
            Time-step durations (seconds). Accepts scalar or [T] vector.
        terminal_reachable_mask : torch.Tensor, optional
            Boolean mask [B, R]. True means hard terminal equality is enforced.
            False switches the reservoir to best-effort terminal attainment.
        terminal_best_effort_target : torch.Tensor, optional
            Terminal storage target [B, R] used when reachable_mask is False.

        Returns
        -------
        q_proj : torch.Tensor
            Projected, feasible outflow [B, T, R].
        V_proj : torch.Tensor
            Storage trajectory derived from q_proj [B, T, R].
        stats : dict
            Projection diagnostics containing residuals and infeasibility mask.
        """

        device = q_raw.device
        dtype = q_raw.dtype
        B, T, R = q_raw.shape

        # Prepare bounds and data tensors for broadcasting
        q_min = self._expand_to_batch(q_min.to(device=device, dtype=dtype), B)
        q_max = self._expand_to_batch(q_max.to(device=device, dtype=dtype), B)
        q_in = q_in.to(device=device, dtype=dtype)

        Vmin = self._expand_to_batch(Vmin.to(device=device, dtype=dtype), B)
        Vmax = self._expand_to_batch(Vmax.to(device=device, dtype=dtype), B)

        V0 = V0.to(device=device, dtype=dtype)
        if V0.dim() == 1:
            V0 = V0.unsqueeze(0).expand(B, -1)
        V_target = V_target.to(device=device, dtype=dtype)
        if V_target.dim() == 1:
            V_target = V_target.unsqueeze(0).expand(B, -1)

        if terminal_reachable_mask is None:
            terminal_reachable_mask = torch.ones((B, R), dtype=torch.bool, device=device)
        else:
            terminal_reachable_mask = terminal_reachable_mask.to(device=device)
            if terminal_reachable_mask.dim() == 1:
                terminal_reachable_mask = terminal_reachable_mask.unsqueeze(0).expand(B, -1)
            elif terminal_reachable_mask.dim() == 2 and terminal_reachable_mask.size(0) == 1 and B > 1:
                terminal_reachable_mask = terminal_reachable_mask.expand(B, -1)
            terminal_reachable_mask = terminal_reachable_mask.to(dtype=torch.bool)

        if terminal_best_effort_target is None:
            terminal_best_effort_target = V_target.clone()
        else:
            terminal_best_effort_target = terminal_best_effort_target.to(device=device, dtype=dtype)
            if terminal_best_effort_target.dim() == 1:
                terminal_best_effort_target = terminal_best_effort_target.unsqueeze(0).expand(B, -1)
            elif terminal_best_effort_target.dim() == 2 and terminal_best_effort_target.size(0) == 1 and B > 1:
                terminal_best_effort_target = terminal_best_effort_target.expand(B, -1)

        delta_t = delta_t.to(device=device, dtype=dtype) if torch.is_tensor(delta_t) else torch.tensor(delta_t, dtype=dtype, device=device)
        if delta_t.dim() == 0:
            delta_t = delta_t.view(1).repeat(T)
        if delta_t.numel() != T:
            raise ValueError(f"delta_t length mismatch: expected {T}, got {delta_t.numel()}")
        dt = delta_t.view(1, T, 1)  # seconds
        dt_vol = dt / self.volume_scale  # convert to storage units (10^8 m^3)

        # 1) Flow box clamp
        q = torch.clamp(q_raw, min=q_min, max=q_max)

        # 2) Compute cumulative storage from clamped flows
        u = (q_in - q) * dt_vol  # incremental storage changes in 10^8 m^3
        w = torch.cumsum(u, dim=1)  # cumulative sum
        w_lower = Vmin - V0.unsqueeze(1)
        w_upper = Vmax - V0.unsqueeze(1)

        w = torch.maximum(w_lower, torch.minimum(w, w_upper))

        final_vmin = Vmin[:, -1, :]
        final_vmax = Vmax[:, -1, :]
        terminal_best_effort_target = torch.maximum(
            final_vmin,
            torch.minimum(terminal_best_effort_target, final_vmax),
        )
        terminal_target_eff = torch.where(
            terminal_reachable_mask,
            V_target,
            terminal_best_effort_target,
        )

        # 3) Redistribute to match terminal targets
        target = terminal_target_eff - V0
        residual = target.unsqueeze(1) - w[:, -1:, :]

        for _ in range(max(self.max_iters, 1)):
            if torch.all(residual.abs() <= self.eps):
                break
            up_slack = (w_upper - w).clamp_min(0.0)
            down_slack = (w - w_lower).clamp_min(0.0)

            total_up = up_slack.sum(dim=1, keepdim=True) + self.eps
            total_down = down_slack.sum(dim=1, keepdim=True) + self.eps

            # Positive residual -> need to add volume (raise storage)
            add_up = (residual.clamp(min=0.0) / total_up) * up_slack
            # Negative residual -> need to remove volume
            add_down = (residual.clamp(max=0.0) / total_down) * down_slack

            w = w + add_up + add_down
            w = torch.maximum(w_lower, torch.minimum(w, w_upper))
            
            # Apply water level ramp constraints if enabled
            # Apply water level ramp constraints if enabled
            if curves is not None and level_limits is not None:
                # Convert cumulative storage to actual storage trajectory
                V_traj = V0.unsqueeze(1) + w  # [B, T, R]
                # Convert storage to level
                H_traj = curves.v2h_all(V_traj)  # [B, T, R]
                
                # Check ramp constraints (only for t >= 1)
                if T > 1:
                    dH_actual = H_traj[:, 1:, :] - H_traj[:, :-1, :]  # [B, T-1, R]
                    
                    # Get ramp limits
                    dH_up_lim = level_limits.get('dH_up')
                    dH_dn_lim = level_limits.get('dH_dn')
                    
                    if dH_up_lim is not None:
                        # Expand to batch if needed
                        if dH_up_lim.dim() == 2:  # [T, R]
                            dH_up_lim = dH_up_lim.unsqueeze(0).expand(B, -1, -1)
                        dH_up_lim_slice = dH_up_lim[:, 1:, :]  # Skip t=0
                        
                        # Check where ramp up exceeds limit
                        exceed_up = dH_actual > dH_up_lim_slice  # [B, T-1, R]
                        
                        if exceed_up.any():
                            # Limit H[t] <= H[t-1] + dH_up
                            H_max_ramp = H_traj[:, :-1, :] + dH_up_lim_slice
                            V_max_ramp = curves.h2v_all(H_max_ramp)
                            w_max_ramp = V_max_ramp - V0.unsqueeze(1)
                            
                            # Tighten upper bound for violating timesteps
                            w_upper[:, 1:, :] = torch.where(
                                exceed_up,
                                torch.minimum(w_upper[:, 1:, :], w_max_ramp),
                                w_upper[:, 1:, :]
                            )
                    
                    if dH_dn_lim is not None:
                        # Expand to batch if needed
                        if dH_dn_lim.dim() == 2:  # [T, R]
                            dH_dn_lim = dH_dn_lim.unsqueeze(0).expand(B, -1, -1)
                        dH_dn_lim_slice = dH_dn_lim[:, 1:, :]
                        
                        # Check where ramp down exceeds limit (dH < -dH_dn)
                        exceed_dn = dH_actual < -dH_dn_lim_slice  # [B, T-1, R]
                        
                        if exceed_dn.any():
                            # Limit H[t] >= H[t-1] - dH_dn
                            H_min_ramp = H_traj[:, :-1, :] - dH_dn_lim_slice
                            V_min_ramp = curves.h2v_all(H_min_ramp)
                            w_min_ramp = V_min_ramp - V0.unsqueeze(1)
                            
                            # Tighten lower bound for violating timesteps
                            w_lower[:, 1:, :] = torch.where(
                                exceed_dn,
                                torch.maximum(w_lower[:, 1:, :], w_min_ramp),
                                w_lower[:, 1:, :]
                            )
                    
                    # Re-apply bounds after tightening
                    w = torch.maximum(w_lower, torch.minimum(w, w_upper))
            
            residual = target.unsqueeze(1) - w[:, -1:, :]

        # Final adjustment on the last time-step within remaining slack
        final_up = (w_upper[:, -1:, :] - w[:, -1:, :]).clamp_min(0.0)
        final_down = (w[:, -1:, :] - w_lower[:, -1:, :]).clamp_min(0.0)
        final_adjust = residual.clone()
        final_adjust = torch.clamp(final_adjust, min=-final_down, max=final_up)
        w[:, -1:, :] = w[:, -1:, :] + final_adjust
        w = torch.maximum(w_lower, torch.minimum(w, w_upper))

        residual = target.unsqueeze(1) - w[:, -1:, :]
        infeasible_mask = (residual.abs() > 5e-4)  # remaining discrepancy beyond tolerance

        # 4) Recover incremental changes and flows
        u_proj = torch.zeros_like(u)
        u_proj[:, 0, :] = w[:, 0, :]
        if T > 1:
            u_proj[:, 1:, :] = w[:, 1:, :] - w[:, :-1, :]

        q_proj = q_in - u_proj / dt_vol
        q_proj = torch.clamp(q_proj, min=q_min, max=q_max)

        # 5) Final storage trajectory (consistent with projected flows)
        V_proj = V0.unsqueeze(1) + torch.cumsum((q_in - q_proj) * dt_vol, dim=1)
        V_proj = torch.maximum(Vmin, torch.minimum(V_proj, Vmax))

        # 6) Optional post-clamp re-balance to restore terminal equality
        if self.post_clamp_rebalance_iters > 0:
            # Recompute w and residual based on clamped flows
            w_pc = V_proj - V0.unsqueeze(1)
            target_pc = terminal_target_eff - V0
            res_pc = target_pc.unsqueeze(1) - w_pc[:, -1:, :]
            # Keep flow/storage copies in sync even if we exit early
            q_pc = q_proj.clone()
            V_pc = V_proj.clone()

            for _ in range(self.post_clamp_rebalance_iters):
                if torch.all(res_pc.abs() <= self.post_clamp_tol):
                    break
                # Slack relative to storage boxes
                w_lower_pc = Vmin - V0.unsqueeze(1)
                w_upper_pc = Vmax - V0.unsqueeze(1)
                up_slack_pc = (w_upper_pc - w_pc).clamp_min(0.0)
                down_slack_pc = (w_pc - w_lower_pc).clamp_min(0.0)
                total_up_pc = up_slack_pc.sum(dim=1, keepdim=True) + self.eps
                total_down_pc = down_slack_pc.sum(dim=1, keepdim=True) + self.eps
                add_up_pc = (res_pc.clamp(min=0.0) / total_up_pc) * up_slack_pc
                add_down_pc = (res_pc.clamp(max=0.0) / total_down_pc) * down_slack_pc
                w_pc = w_pc + add_up_pc + add_down_pc
                w_pc = torch.maximum(w_lower_pc, torch.minimum(w_pc, w_upper_pc))
                res_pc = target_pc.unsqueeze(1) - w_pc[:, -1:, :]

                # Project back to flows (respect flow boxes)
                u_pc = torch.zeros_like(u)
                u_pc[:, 0, :] = w_pc[:, 0, :]
                if T > 1:
                    u_pc[:, 1:, :] = w_pc[:, 1:, :] - w_pc[:, :-1, :]
                q_pc = q_in - u_pc / dt_vol
                q_pc = torch.clamp(q_pc, min=q_min, max=q_max)
                # Recompute storage trajectory for next iteration
                V_pc = V0.unsqueeze(1) + torch.cumsum((q_in - q_pc) * dt_vol, dim=1)
                V_pc = torch.maximum(Vmin, torch.minimum(V_pc, Vmax))
                w_pc = V_pc - V0.unsqueeze(1)
                res_pc = target_pc.unsqueeze(1) - w_pc[:, -1:, :]

            # Final direct flow-space redistribution along available q slack
            need_final = (target_pc - w_pc[:, -1, :])  # [B,R]
            if torch.any(need_final.abs() > self.post_clamp_tol):
                # Positive need -> reduce q where (q_pc - q_min) > 0; Negative need -> increase q where (q_max - q_pc) > 0
                red_slack = ((q_pc - q_min).clamp_min(0.0) * dt_vol).sum(dim=1) + self.eps  # [B,R]
                inc_slack = (((q_max - q_pc).clamp_min(0.0)) * dt_vol).sum(dim=1) + self.eps  # [B,R]
                pos_mask = (need_final > 0.0).float()
                neg_mask = (need_final < 0.0).float()
                # Volume to adjust per reservoir
                vol_pos = (need_final * pos_mask)  # [B,R]
                vol_neg = (need_final * neg_mask)  # [B,R] (negative)
                # Distribute across time by proportional slack
                red_weights = (q_pc - q_min).clamp_min(0.0) * dt_vol  # [B,T,R]
                red_weights_sum = red_weights.sum(dim=1, keepdim=True) + self.eps
                red_weights = red_weights / red_weights_sum
                inc_weights = (q_max - q_pc).clamp_min(0.0) * dt_vol
                inc_weights_sum = inc_weights.sum(dim=1, keepdim=True) + self.eps
                inc_weights = inc_weights / inc_weights_sum
                # Apply volume adjustments -> delta_q = delta_V * weight / dt
                delta_q_pos = (vol_pos.unsqueeze(1) * red_weights) / dt_vol  # reduce q
                delta_q_neg = (vol_neg.unsqueeze(1) * inc_weights) / dt_vol  # increase q (neg volume)
                q_pc = q_pc - delta_q_pos + delta_q_neg
                q_pc = torch.clamp(q_pc, min=q_min, max=q_max)
                V_pc = V0.unsqueeze(1) + torch.cumsum((q_in - q_pc) * dt_vol, dim=1)
                V_pc = torch.maximum(Vmin, torch.minimum(V_pc, Vmax))
                w_pc = V_pc - V0.unsqueeze(1)
                res_pc = target_pc.unsqueeze(1) - w_pc[:, -1:, :]

            # Commit post-clamp refined solution
            q_proj = q_pc
            V_proj = V_pc
            residual = res_pc  # for stats
            infeasible_mask = (res_pc.abs() > self.post_clamp_tol)
        else:
            # residual based on pre-clamp balance; recompute using final V_proj for stats
            residual = target.unsqueeze(1) - (V_proj - V0.unsqueeze(1))[:, -1:, :]
            infeasible_mask = (residual.abs() > self.post_clamp_tol)

        if self.terminal_second_rebalance_iters > 0 and T > 1:
            base_tail = min(self.terminal_second_rebalance_window, T)
            extra_tail = self.tail_focus_extra_window if self.tail_focus_indices else 0
            K_tail = min(base_tail + extra_tail, T)
            if K_tail > 1:
                tail_start = T - K_tail
                focus_mask = None
                fraction_vec = None
                decay_weights = None
                if self.tail_focus_indices:
                    focus_mask = torch.zeros((1, 1, R), device=device, dtype=q_proj.dtype)
                    focus_indices = list(self.tail_focus_indices)
                    focus_mask[..., focus_indices] = 1.0
                    if self.tail_focus_step_fraction < 1.0:
                        frac = torch.ones((1, R), device=device, dtype=q_proj.dtype)
                        frac[..., focus_indices] = self.tail_focus_step_fraction
                        fraction_vec = frac
                    if self.tail_focus_time_decay > 0.0 and K_tail > 1:
                        decay_vec = torch.linspace(
                            1.0 + self.tail_focus_time_decay,
                            1.0,
                            steps=K_tail,
                            device=device,
                            dtype=q_proj.dtype,
                        ).view(1, -1, 1)
                        decay_weights = 1.0 + focus_mask * (decay_vec - 1.0)
                for _ in range(self.terminal_second_rebalance_iters):
                    need = terminal_target_eff - V_proj[:, -1, :]
                    if torch.all(need.abs() <= self.post_clamp_tol):
                        break
                    q_tail = q_proj[:, tail_start:, :]
                    q_min_tail = q_min[:, tail_start:, :]
                    q_max_tail = q_max[:, tail_start:, :]
                    dt_tail = dt_vol[:, tail_start:, :].expand(B, -1, -1)
                    need_pos = need.clamp_min(0.0)
                    if fraction_vec is not None:
                        need_pos = need_pos * fraction_vec
                    if torch.any(need_pos > self.post_clamp_tol):
                        # Reduce tail outflow directly to increase terminal storage.
                        # Do not compensate at the last period; otherwise total volume is unchanged
                        # and terminal storage cannot improve.
                        slack_down = (q_tail - q_min_tail).clamp_min(0.0) * dt_tail
                        if decay_weights is not None:
                            slack_down = slack_down * decay_weights
                        vol_down = slack_down.sum(dim=1, keepdim=True)
                        mask_down = vol_down > self.eps
                        adj_down = torch.zeros_like(slack_down)
                        if torch.any(mask_down):
                            mask_down_exp = mask_down.expand_as(slack_down)
                            adj_down = torch.where(
                                mask_down_exp,
                                need_pos.unsqueeze(1) * slack_down / vol_down.clamp_min(self.eps),
                                torch.zeros_like(slack_down),
                            )
                        q_tail[...] = q_tail - adj_down / dt_tail
                    need_neg = need.clamp_max(0.0)
                    if fraction_vec is not None:
                        need_neg = need_neg * fraction_vec
                    if torch.any(need_neg < -self.post_clamp_tol):
                        # Increase tail outflow directly to lower terminal storage.
                        slack_up = (q_max_tail - q_tail).clamp_min(0.0) * dt_tail
                        if decay_weights is not None:
                            slack_up = slack_up * decay_weights
                        vol_up = slack_up.sum(dim=1, keepdim=True)
                        mask_up = vol_up > self.eps
                        adj_up = torch.zeros_like(slack_up)
                        if torch.any(mask_up):
                            mask_up_exp = mask_up.expand_as(slack_up)
                            adj_up = torch.where(
                                mask_up_exp,
                                (-need_neg).unsqueeze(1) * slack_up / vol_up.clamp_min(self.eps),
                                torch.zeros_like(slack_up),
                            )
                        q_tail[...] = q_tail + adj_up / dt_tail
                    q_proj = torch.clamp(q_proj, min=q_min, max=q_max)
                    V_proj = V0.unsqueeze(1) + torch.cumsum((q_in - q_proj) * dt_vol, dim=1)
                    V_proj = torch.maximum(Vmin, torch.minimum(V_proj, Vmax))
                residual = target.unsqueeze(1) - (V_proj - V0.unsqueeze(1))[:, -1:, :]
                infeasible_mask = (residual.abs() > self.post_clamp_tol)

        requested_residual = V_target.unsqueeze(1) - V_proj[:, -1:, :]
        stats: Dict[str, torch.Tensor] = {
            "residual": residual.squeeze(1).detach(),
            "requested_residual": requested_residual.squeeze(1).detach(),
            "infeasible": infeasible_mask.squeeze(1).detach(),
            "terminal_hard_enforced": terminal_reachable_mask.detach(),
            "terminal_requested_target": V_target.detach(),
            "terminal_effective_target": terminal_target_eff.detach(),
        }
        return q_proj, V_proj, stats
