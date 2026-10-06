"""Minimal conditional flow-matching model for trajectory experiments.

This module is intentionally small and auditable. It provides the core vector-field
objective; experiment orchestration, calibration, and solver verification live in
run_experiment.py. Install torch before running it.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class FlowMatchingConfig:
    state_dim: int = 3
    horizon: int = 32
    condition_dim: int = 3
    hidden_dim: int = 256
    time_dim: int = 32

class FourierTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half = max(1, dim // 2)
        self.register_buffer("freq", torch.exp(torch.linspace(0.0, 5.0, half)))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        phase = t[..., None] * self.freq
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

class ConditionalVectorField(nn.Module):
    def __init__(self, config: FlowMatchingConfig):
        super().__init__()
        self.config = config
        flat = config.state_dim * config.horizon
        self.time = FourierTimeEmbedding(config.time_dim)
        self.net = nn.Sequential(
            nn.Linear(flat + config.condition_dim + config.time_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, flat),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z.flatten(1), condition, self.time(t)], dim=-1)).view_as(z)

def conditional_flow_matching_loss(model: nn.Module, target: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    z0 = torch.randn_like(target)
    z1 = target
    t = torch.rand(target.shape[0], device=target.device)
    t_view = t.view(-1, *([1] * (target.ndim - 1)))
    zt = (1.0 - t_view) * z0 + t_view * z1
    velocity = z1 - z0
    prediction = model(zt, t, condition)
    return torch.mean((prediction - velocity) ** 2)

def sample_ode(model: nn.Module, condition: torch.Tensor, shape: tuple[int, ...], steps: int = 64) -> torch.Tensor:
    z = torch.randn(shape, device=condition.device)
    dt = 1.0 / steps
    with torch.no_grad():
        for i in range(steps):
            t = torch.full((shape[0],), i / steps, device=condition.device)
            z = z + dt * model(z, t, condition)
    return z
