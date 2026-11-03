"""DDPM noise scheduler with cosine beta schedule."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _cosine_betas(num_timesteps: int, max_beta: float = 0.999) -> torch.Tensor:
  """Nichol & Dhariwal cosine β schedule with ``s = 0.008``.

      ᾱ(t) = cos²((t + 0.008) / 1.008 · π/2),  t ∈ [0, 1]
      β_i = min(1 − ᾱ((i+1)/T) / ᾱ(i/T), max_beta)

  Reaches ᾱ_T ≈ 0 even for small T (e.g. T=50), unlike the standard linear
  schedule which only works at T≈1000.
  """

  def alpha_bar(t: float) -> float:
    return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

  betas = []
  for i in range(num_timesteps):
    t1 = i / num_timesteps
    t2 = (i + 1) / num_timesteps
    betas.append(min(1.0 - alpha_bar(t2) / alpha_bar(t1), max_beta))
  return torch.tensor(betas, dtype=torch.float32)


class DDPMScheduler(nn.Module):
  """Minimal DDPM noise scheduler with a cosine beta schedule.

  Subclassing nn.Module so buffers move with .to(device) automatically.
  """

  sqrt_alphas_cumprod: torch.Tensor
  sqrt_one_minus_alphas_cumprod: torch.Tensor
  betas: torch.Tensor
  alphas_cumprod: torch.Tensor
  alphas_cumprod_prev: torch.Tensor
  sqrt_recip_alphas_cumprod: torch.Tensor
  sqrt_recipm1_alphas_cumprod: torch.Tensor
  posterior_variance: torch.Tensor
  posterior_mean_coef1: torch.Tensor
  posterior_mean_coef2: torch.Tensor

  def __init__(
    self,
    num_timesteps: int = 50,
  ) -> None:
    super().__init__()
    self.num_timesteps = num_timesteps
    betas = _cosine_betas(num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat(
      [torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0
    )
    self.register_buffer("betas", betas)
    self.register_buffer("alphas_cumprod", alphas_cumprod)
    self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
    self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
    self.register_buffer(
      "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
    )
    self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
    self.register_buffer(
      "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0)
    )
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    self.register_buffer("posterior_variance", posterior_variance)
    self.register_buffer(
      "posterior_mean_coef1",
      betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
    )
    self.register_buffer(
      "posterior_mean_coef2",
      (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
    )

  def step(self, eps: torch.Tensor, x_t: torch.Tensor, t: int) -> torch.Tensor:
    """One ancestral DDPM denoising step: x_t -> x_{t-1}.

    Predicts x_0 from eps, computes the posterior mean, and adds noise (except
    when t == 0). Operates on a single integer timestep broadcast across batch.
    """
    x_0_hat = (
      self.sqrt_recip_alphas_cumprod[t] * x_t
      - self.sqrt_recipm1_alphas_cumprod[t] * eps
    )
    mean = self.posterior_mean_coef1[t] * x_0_hat + self.posterior_mean_coef2[t] * x_t
    if t == 0:
      return mean
    noise = torch.randn_like(x_t)
    return mean + torch.sqrt(self.posterior_variance[t]) * noise

  def add_noise(
    self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
  ) -> torch.Tensor:
    """Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise."""
    shape = (-1, *([1] * (x_0.ndim - 1)))
    return (
      self.sqrt_alphas_cumprod[t].view(shape) * x_0
      + self.sqrt_one_minus_alphas_cumprod[t].view(shape) * noise
    )

  def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(
      0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long
    )
