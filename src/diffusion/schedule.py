"""Cosine noise schedule and the x0/eps/v conversions, for a 64-d latent.

Cosine (Nichol & Dhariwal) rather than linear: the standard linear beta was tuned
for 3x32x32+ images, and at d=64 it destroys signal far too early. The paper uses
cosine with 1000 timesteps, which is what this matches.
"""
import math

import torch

PARAMETERIZATIONS = ("x0", "eps", "v")


def cosine_alpha_bar(timesteps: int, s: float = 0.008) -> torch.Tensor:
    t = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    return alpha_bar.clamp(1e-8, 1.0)


class NoiseSchedule:
    def __init__(self, timesteps: int = 1000, s: float = 0.008, device=None):
        self.timesteps = timesteps
        ab = cosine_alpha_bar(timesteps, s).to(torch.float32)
        self.alpha_bar = ab[1:].to(device)                 # (T,), index t-1 -> t
        self.alpha_bar_prev = ab[:-1].to(device)
        self.sqrt_ab = self.alpha_bar.sqrt()
        self.sqrt_1mab = (1.0 - self.alpha_bar).sqrt()

    def to(self, device):
        for k in ("alpha_bar", "alpha_bar_prev", "sqrt_ab", "sqrt_1mab"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        return self.sqrt_ab[t].unsqueeze(-1) * x0 + self.sqrt_1mab[t].unsqueeze(-1) * noise

    def target_for(self, parameterization: str, x0: torch.Tensor, noise: torch.Tensor,
                   t: torch.Tensor) -> torch.Tensor:
        if parameterization == "x0":
            return x0
        if parameterization == "eps":
            return noise
        if parameterization == "v":
            return self.sqrt_ab[t].unsqueeze(-1) * noise - self.sqrt_1mab[t].unsqueeze(-1) * x0
        raise ValueError(f"unknown parameterization {parameterization!r}")

    def to_x0(self, pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor,
              parameterization: str) -> torch.Tensor:
        a, sm = self.sqrt_ab[t].unsqueeze(-1), self.sqrt_1mab[t].unsqueeze(-1)
        if parameterization == "x0":
            return pred
        if parameterization == "eps":
            return (x_t - sm * pred) / a
        if parameterization == "v":
            return a * x_t - sm * pred
        raise ValueError(f"unknown parameterization {parameterization!r}")

    def to_eps(self, x0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        a, sm = self.sqrt_ab[t].unsqueeze(-1), self.sqrt_1mab[t].unsqueeze(-1)
        return (x_t - a * x0) / sm


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb
