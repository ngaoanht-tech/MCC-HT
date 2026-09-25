"""Differentiable joint projection for the final cascade release sequence.

This layer follows the existing model and level projection in both training
and inference. Water-level change constraints remain in the existing code.
"""

from __future__ import annotations

from functools import lru_cache

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer


FLOW_SCALE = 1e4  # m3/s
VOLUME_SCALE = 1e2  # 1e8 m3


@lru_cache(maxsize=16)
def _layer(
    periods: int,
    reservoirs: int,
    seconds: tuple[float, ...],
    ramp_up: tuple[float, ...],
    ramp_down: tuple[float, ...],
) -> CvxpyLayer:
    """Construct one DPP-compliant quadratic projection, cached by shape."""
    T, R = periods, reservoirs
    q = cp.Variable((T, R))  # flow / FLOW_SCALE
    v = cp.Variable((T, R))  # storage / VOLUME_SCALE
    proposed = cp.Parameter((T, R))
    head = cp.Parameter(T)
    interval = cp.Parameter((T, R - 1))
    initial = cp.Parameter(R)
    terminal = cp.Parameter(R)
    qmin = cp.Parameter((T, R))
    qmax = cp.Parameter((T, R))
    vmin = cp.Parameter((T, R))
    vmax = cp.Parameter((T, R))

    constraints = [q >= qmin, q <= qmax, v >= vmin, v <= vmax, v[T - 1, :] == terminal]
    for t in range(T):
        # q and external inflows are both in FLOW_SCALE units; v in VOLUME_SCALE.
        coefficient = seconds[t] * FLOW_SCALE / (1e8 * VOLUME_SCALE)
        for r in range(R):
            previous = initial[r] if t == 0 else v[t - 1, r]
            inflow = head[t] if r == 0 else q[t, r - 1] + interval[t, r - 1]
            constraints.append(v[t, r] == previous + coefficient * (inflow - q[t, r]))

    if ramp_up and ramp_down and T > 1:
        up = np.asarray(ramp_up, dtype=np.float64).reshape(R, T - 1).T / FLOW_SCALE
        down = np.asarray(ramp_down, dtype=np.float64).reshape(R, T - 1).T / FLOW_SCALE
        constraints.extend([q[1:, :] - q[:-1, :] <= up, q[:-1, :] - q[1:, :] <= down])

    problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(q - proposed)), constraints)
    if not problem.is_dpp():
        raise RuntimeError("joint projection problem must be DPP-compliant")
    return CvxpyLayer(
        problem,
        parameters=[proposed, head, interval, initial, terminal, qmin, qmax, vmin, vmax],
        variables=[q, v],
    )


def project_joint_schedule(
    q_proposed: torch.Tensor,
    head_inflow: torch.Tensor,
    interval_inflow: torch.Tensor | None,
    V0: torch.Tensor,
    V_target: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
    V_min: torch.Tensor,
    V_max: torch.Tensor,
    delta_t: torch.Tensor,
    flow_ramp_up: list[list[float]] | None = None,
    flow_ramp_down: list[list[float]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project [B,T,R] releases and return (release, storage) with gradients."""
    if q_proposed.ndim != 3:
        raise ValueError("q_proposed must have shape [B,T,R]")
    B, T, R = q_proposed.shape
    if R < 2:
        raise ValueError("joint cascade layer expects at least two reservoirs")
    if interval_inflow is None:
        interval_inflow = q_proposed.new_zeros((B, T, R - 1))

    def batch_time(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return x.expand(B, -1, -1)

    def batch_vector(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return x.expand(B, -1)

    seconds = tuple(float(x) for x in delta_t.detach().cpu().reshape(-1).tolist())
    if len(seconds) != T or any(x <= 0 for x in seconds):
        raise ValueError("delta_t must contain one positive duration per period")
    up = tuple(float(x) for row in (flow_ramp_up or []) for x in row[: max(0, T - 1)])
    down = tuple(float(x) for row in (flow_ramp_down or []) for x in row[: max(0, T - 1)])
    if bool(up) != bool(down) or (up and (len(up) != R * (T - 1) or len(down) != R * (T - 1))):
        raise ValueError("flow ramp limits must have shape [R,T-1]")
    solver = _layer(T, R, seconds, up, down)

    # Double precision keeps the balance and terminal equalities accurate;
    # converting back to the model dtype preserves autograd to q_proposed.
    dtype = q_proposed.dtype
    device = q_proposed.device
    def to_double(x: torch.Tensor) -> torch.Tensor:
        return x.to(device=device, dtype=torch.float64)

    q_solution, v_solution = solver(
        to_double(q_proposed) / FLOW_SCALE,
        to_double(batch_vector(head_inflow)) / FLOW_SCALE,
        to_double(batch_time(interval_inflow)) / FLOW_SCALE,
        to_double(batch_vector(V0)) / VOLUME_SCALE,
        to_double(batch_vector(V_target)) / VOLUME_SCALE,
        to_double(batch_time(q_min)) / FLOW_SCALE,
        to_double(batch_time(q_max)) / FLOW_SCALE,
        to_double(batch_time(V_min)) / VOLUME_SCALE,
        to_double(batch_time(V_max)) / VOLUME_SCALE,
        solver_args={"eps": 1e-8, "max_iters": 10000},
    )
    return (q_solution * FLOW_SCALE).to(dtype), (v_solution * VOLUME_SCALE).to(dtype)
