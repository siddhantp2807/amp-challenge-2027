"""Training loss and DDIM sampling, including classifier-free guidance and CADS."""
import numpy as np
import torch

from src.diffusion.conditioning import PROPERTIES, sample_masks
from src.diffusion.schedule import NoiseSchedule


class LatentDiffusion:
    def __init__(self, model, schedule: NoiseSchedule, parameterization: str = "x0",
                 self_cond: bool = False):
        self.model = model
        self.schedule = schedule
        self.parameterization = parameterization
        self.self_cond = self_cond

    # ---- training ----------------------------------------------------------

    def loss(self, z0: torch.Tensor, c: torch.Tensor, is_target: torch.Tensor,
             generator: torch.Generator | None = None) -> torch.Tensor:
        b = z0.shape[0]
        device = z0.device
        t = torch.randint(0, self.schedule.timesteps, (b,), device=device)
        noise = torch.randn_like(z0)
        x_t = self.schedule.q_sample(z0, t, noise)
        mask = sample_masks(b, generator).to(device)

        x0_prev = None
        if self.self_cond and torch.rand(()) < 0.5:
            with torch.no_grad():
                pred = self.model(x_t, t, c, mask, is_target)
                x0_prev = self.schedule.to_x0(pred, x_t, t, self.parameterization).detach()

        pred = self.model(x_t, t, c, mask, is_target, x0_prev=x0_prev)
        target = self.schedule.target_for(self.parameterization, z0, noise, t)
        return torch.nn.functional.mse_loss(pred, target)

    @torch.no_grad()
    def eval_loss(self, z0, c, is_target, seed: int = 0) -> float:
        """Deterministic val loss: fixed timesteps/noise/masks so epoch-to-epoch
        movement is model change, not resampling noise."""
        device = z0.device
        g = torch.Generator(device="cpu").manual_seed(seed)
        b = z0.shape[0]
        t = torch.randint(0, self.schedule.timesteps, (b,), generator=g).to(device)
        noise = torch.randn(z0.shape, generator=g).to(device)
        x_t = self.schedule.q_sample(z0, t, noise)
        mask = sample_masks(b, g).to(device)
        pred = self.model(x_t, t, c, mask, is_target)
        target = self.schedule.target_for(self.parameterization, z0, noise, t)
        return torch.nn.functional.mse_loss(pred, target).item()

    # ---- sampling ----------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(self, c: torch.Tensor, mask: torch.Tensor, is_target: torch.Tensor,
                    steps: int = 50, cfg_weight: float = 1.0, eta: float = 0.0,
                    cads: dict | None = None, generator: torch.Generator | None = None,
                    x_T: torch.Tensor | None = None) -> torch.Tensor:
        """Reverse DDIM from noise to z0.

        cfg_weight = 1.0 is plain conditional sampling. Above that, guidance uses
        the all-masked (unconditional) model, which the training mask distribution
        trains on 25% of steps -- so CFG needs no separate unconditional pass at
        training time.

        `cads` enables condition annealing (Sadat et al.). The paper anneals a
        continuous condition vector, but our length branch is a discrete embedding
        table and the continuous ones go through Fourier features, where adding
        Gaussian noise is not the same operation as perturbing the value. So the
        default mode anneals the *mask* instead: at high t each property is masked
        with probability 1 - gamma(t), reusing the already-trained mask tokens.
        """
        device = next(self.model.parameters()).device
        b = c.shape[0]
        x = (torch.randn(b, self.model.d_z, generator=generator).to(device)
             if x_T is None else x_T.to(device))
        null_mask = torch.zeros_like(mask)

        ts = torch.linspace(self.schedule.timesteps - 1, 0, steps).long().to(device)
        x0_prev = None
        for i, t_scalar in enumerate(ts):
            t = t_scalar.expand(b)
            step_mask, step_c = mask, c
            if cads is not None:
                step_mask, step_c = self._cads(c, mask, t_scalar.item(), cads, generator)

            pred = self.model(x, t, step_c, step_mask, is_target, x0_prev=x0_prev)
            x0 = self.schedule.to_x0(pred, x, t, self.parameterization)
            if cfg_weight != 1.0:
                pred_u = self.model(x, t, step_c, null_mask, is_target, x0_prev=x0_prev)
                x0_u = self.schedule.to_x0(pred_u, x, t, self.parameterization)
                x0 = x0_u + cfg_weight * (x0 - x0_u)
            x0_prev = x0 if self.self_cond else None

            eps = self.schedule.to_eps(x0, x, t)
            if i == len(ts) - 1:
                x = x0
                break
            ab_next = self.schedule.alpha_bar[ts[i + 1]].unsqueeze(-1)
            sigma = eta * ((1 - ab_next) / (1 - self.schedule.alpha_bar[t].unsqueeze(-1))
                           * (1 - self.schedule.alpha_bar[t].unsqueeze(-1) / ab_next)).clamp(min=0).sqrt()
            x = ab_next.sqrt() * x0 + (1 - ab_next - sigma ** 2).clamp(min=0).sqrt() * eps
            if eta > 0:
                x = x + sigma * torch.randn(x.shape, generator=generator).to(device)
        return x

    def _cads(self, c, mask, t_scalar, cfg, generator):
        """gamma(t): 1 below tau1*T, 0 above tau2*T, linear in between."""
        tau1, tau2 = cfg.get("tau1", 0.5), cfg.get("tau2", 0.9)
        frac = t_scalar / max(self.schedule.timesteps - 1, 1)
        if frac <= tau1:
            gamma = 1.0
        elif frac >= tau2:
            gamma = 0.0
        else:
            gamma = (tau2 - frac) / (tau2 - tau1)

        if cfg.get("mode", "mask") == "mask":
            keep = torch.rand(mask.shape, generator=generator).to(mask.device) < gamma
            return mask & keep, c
        # "value" mode: noise the standardized scalars before the embedders
        s = cfg.get("noise_scale", 0.1)
        noise = torch.randn(c.shape, generator=generator).to(c.device)
        return mask, (gamma ** 0.5) * c + s * ((1 - gamma) ** 0.5) * noise


class EMA:
    """Exponential moving average of weights; evaluate this copy, not the live one."""

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)

    def state_dict(self):
        return self.shadow
