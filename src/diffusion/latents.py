"""Encoding sequences to stage-1 latents, and the standardization stage 2 uses.

The VAE is loaded strictly read-only. In particular this never runs
src/freeze/export_frozen.py: SUMMARY.md recommends prototyping stage 2 on
finetune_dz64_fb2p5_v1_best *without* freezing, and encoding needs no freeze.

That leaves a trap worth stating. SequenceVAE carries latent_mean/latent_std
buffers that export_frozen.py would populate, and SequenceVAE.normalize() would
then apply them. Stage 2 computes its *own* z_stats over its own training split
and stores them in the diffusion checkpoint. If someone later freezes the VAE,
the two normalizations would silently compose, so assert_vae_unfrozen() fails
loudly instead.
"""
import numpy as np
import torch

from src.eval.common import encode_sequences
from src.eval.freeze_report import build_model_from_checkpoint


def load_vae(checkpoint: str, device):
    model = build_model_from_checkpoint(checkpoint, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    assert_vae_unfrozen(model, checkpoint)
    return model


def assert_vae_unfrozen(model, checkpoint: str) -> None:
    """Stage 2 owns the latent normalization; the VAE's own must still be identity."""
    if not (torch.allclose(model.latent_mean, torch.zeros_like(model.latent_mean))
            and torch.allclose(model.latent_std, torch.ones_like(model.latent_std))):
        raise RuntimeError(
            f"{checkpoint} has non-identity latent_mean/latent_std, so export_frozen.py "
            "has been run on it. Stage 2 applies its own z_stats; using this checkpoint "
            "would double-normalize. Point --vae-checkpoint at the unfrozen checkpoint, "
            "or retrain stage 2 against the frozen one."
        )


@torch.no_grad()
def encode(model, sequences: list[str], device, sample: bool = False,
           batch_size: int = 256) -> torch.Tensor:
    return encode_sequences(model, sequences, device, batch_size=batch_size, sample=sample)


@torch.no_grad()
def encode_posterior(model, sequences: list[str], device,
                     batch_size: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """(mu, logvar) for every sequence.

    Training draws z ~ q(z|x) fresh each epoch rather than using mu: the encoder's
    own logvar is a calibrated statement of how far z can move without changing
    the sequence, so this is the VAE's own augmentation rather than arbitrary
    jitter. Labels come from the sequence, so resampling teaches the model the
    conditional *spread* p(z|c) instead of a single point per condition.
    """
    from src.data.dataset import collate_fn
    from src.data.tokenizer import VOCAB

    model.eval()
    encoded = [VOCAB.encode(s) for s in sequences]
    mus, logvars = [], []
    for i in range(0, len(encoded), batch_size):
        batch = collate_fn(encoded[i : i + batch_size])
        mu, logvar = model.encode(batch["tokens"].to(device), batch["pad_mask"].to(device))
        mus.append(mu.cpu())
        logvars.append(logvar.cpu())
    return torch.cat(mus), torch.cat(logvars)


class LatentStats:
    """Diagonal mean/std over the stage-2 training latents.

    Diagonal, not ZCA: measured per-dim std is 0.783-1.000, so whitening buys
    almost nothing on the schedule side while rotating the latent out of the
    decoder's native basis -- which mixes its well-behaved and degenerate
    directions and makes per-dim diagnostics uninterpretable.

    Recomputed per training phase (the 31k union and the 6.7k target set have
    different statistics) and stored in that phase's checkpoint.
    """

    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32).clamp(min=1e-6)

    @classmethod
    def fit(cls, z: torch.Tensor) -> "LatentStats":
        return cls(z.mean(0), z.std(0))

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.mean.to(z.device)) / self.std.to(z.device)

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std.to(z.device) + self.mean.to(z.device)

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, d: dict) -> "LatentStats":
        return cls(torch.tensor(d["mean"]), torch.tensor(d["std"]))


def posterior_width_report(logvar: torch.Tensor, marginal_std: torch.Tensor) -> dict:
    """G6: is z-resampling real augmentation or cosmetic?

    If the mean posterior std is tiny relative to the marginal spread of mu, then
    sampling z ~ q(z|x) barely moves anything and an explicit jitter sweep is
    needed instead.
    """
    post_std = torch.exp(0.5 * logvar)
    ratio = (post_std.mean(0) / marginal_std).mean().item()
    return {
        "mean_posterior_std": post_std.mean().item(),
        "mean_marginal_std": marginal_std.mean().item(),
        "ratio": ratio,
        "augmentation_is_meaningful": bool(ratio >= 0.15),
    }
