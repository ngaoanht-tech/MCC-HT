"""Physical trajectory checks for the paper's outer stopping criterion."""
import torch


def cascade_state(q, head, interval, initial, seconds):
    if q.ndim != 3:
        raise ValueError('Expected [batch, period, reservoir] releases.')
    incoming = torch.zeros_like(q)
    incoming[..., 0] = head[..., 0] if head.ndim == 3 else head
    if q.size(-1) > 1:
        incoming[..., 1:] = q[..., :-1]
        if interval is not None:
            incoming[..., 1:] += interval
    storage = initial.unsqueeze(1) + torch.cumsum(
        (incoming - q) * seconds.reshape(1, -1, 1) / 1e8, dim=1)
    return incoming, storage


@torch.no_grad()
def operating_checks(q, head, interval, initial, target, seconds, qmin, qmax,
                     vmin, vmax, curves, level_limits=None, flow_ramps=None,
                     capacities=None):
    """Check reconstructed trajectories, never a separately clipped storage.

    Path bounds have no extra tolerance. Terminal water-level tolerance is
    0.01 m. Return the individual checks rather than inferring feasibility
    from a small change between iterations.
    """
    if curves is None:
        raise ValueError('Water-level curves are required for the 0.01 m terminal check.')
    q = q.detach().double()
    initial, target = initial.double(), target.double()
    incoming, storage = cascade_state(q, head.double(),
        None if interval is None else interval.double(), initial, seconds.double())
    h = curves.v2h_all(storage)
    ht = curves.v2h_all(target)
    checks = dict(finite=bool(torch.isfinite(q).all() & torch.isfinite(storage).all()),
                  release=bool(((q >= qmin) & (q <= qmax)).all()),
                  storage=bool(((storage >= vmin) & (storage <= vmax)).all()),
                  terminal=bool(((h[:, -1] - ht).abs() <= .01).all()))
    if level_limits is not None:
        checks['water_level'] = bool(((h >= level_limits['H_min']) &
                                     (h <= level_limits['H_max'])).all())
        h0 = curves.v2h_all(initial).unsqueeze(1)
        dh = torch.diff(torch.cat([h0, h], dim=1), dim=1)
        if 'dH_up' in level_limits and 'dH_dn' in level_limits:
            checks['water_level_ramp'] = bool(((dh <= level_limits['dH_up']) &
                                              (-dh <= level_limits['dH_dn'])).all())
    if flow_ramps and q.size(1) > 1:
        diff = torch.diff(q, dim=1)
        for key, sign in (('ramp_up', 1), ('ramp_down', -1)):
            if flow_ramps.get(key) is not None:
                bound = torch.as_tensor(flow_ramps[key], device=q.device, dtype=q.dtype)
                if bound.ndim == 2:
                    bound = bound.T[:q.size(1)-1, :q.size(2)]
                checks[key] = bool((sign * diff <= bound).all())
    if capacities is not None:
        tail = curves.q2zt_all(q)
        power = 1000. * 9.81 * .88 * q * (h - tail).clamp_min(0.) / 1e6
        cap = torch.as_tensor(capacities, device=q.device, dtype=q.dtype)
        power = torch.minimum(power.clamp_min(0.), cap)
        checks['power'] = bool(torch.isfinite(power).all() & ((power >= 0) & (power <= cap)).all())
    return dict(checks=checks, feasible=all(checks.values()),
                storage=storage, levels=h, inflows=incoming)
