import torch

from constraint.differentiable_joint_projection import project_joint_schedule


def test_joint_projection_balances_cascade_and_propagates_gradient():
    proposed = torch.tensor(
        [[[9.0, 11.0], [11.0, 9.0], [10.0, 10.0]]], requires_grad=True
    )
    head = torch.full((1, 3), 10.0)
    interval = torch.zeros((1, 3, 1))
    initial = torch.tensor([[5.0, 5.0]])
    target = initial.clone()
    qmin = torch.zeros((1, 3, 2))
    qmax = torch.full((1, 3, 2), 20.0)
    vmin = torch.zeros((3, 2))
    vmax = torch.full((3, 2), 10.0)
    dt = torch.full((3,), 1e6)

    q, v = project_joint_schedule(
        proposed, head, interval, initial, target, qmin, qmax, vmin, vmax, dt
    )
    inflow = torch.cat((head.unsqueeze(-1), q[..., :1] + interval), dim=-1)
    calculated = initial.unsqueeze(1) + torch.cumsum((inflow - q) * dt.view(1, -1, 1) / 1e8, dim=1)
    assert torch.allclose(v, calculated, atol=1e-4)
    assert torch.allclose(v[:, -1], target, atol=1e-4)
    q[0, 0, 0].backward()
    assert proposed.grad is not None
    assert torch.isfinite(proposed.grad).all()
    assert proposed.grad.abs().sum() > 0
