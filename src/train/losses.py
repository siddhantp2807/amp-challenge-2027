"""Loss components for the stage-1 VAE, with two smoothness-oriented additions
beyond a plain beta-VAE ELBO:

- free bits: floors each latent dim's KL contribution so beta-annealing alone
  can't let dims re-collapse to the prior once they saturate ("dead dims"
  waste diffusion-model capacity and add unmodeled noise).
- a Lipschitz/consistency penalty: decode z and a small perturbation of z,
  penalize disproportionate output divergence. This trains directly for
  "nearby z -> nearby decoded sequence" rather than only checking for it
  post-hoc in eval/posterior_collapse.py.

There is also a length-prediction auxiliary loss: the non-AR decoder is trained
with bucketed batches padded only to each batch's own bucket max, so it never
gets gradient for what to emit past a given batch's length range out to the
full max_len. Rather than trying to fix that training signal directly (which
would reintroduce the pad-domination problem bucketing avoids), a small head
predicts sequence length from z, and eval/generation decode truncates to that
predicted length instead of asking the decoder to self-terminate correctly at
positions it was rarely trained on.
"""
import torch
import torch.nn.functional as F

from src.data.tokenizer import PAD_ID
from src.model.vae import SequenceVAE


def reconstruction_loss(
    logits: torch.Tensor, targets: torch.Tensor, pad_mask: torch.Tensor
) -> torch.Tensor:
    """Per-position CE, masked, non-pad-token mean (not sequence-mean) so
    short sequences aren't implicitly upweighted relative to long ones.
    """
    B, L, V = logits.shape
    loss_per_tok = F.cross_entropy(
        logits.reshape(B * L, V), targets.reshape(B * L), reduction="none"
    ).reshape(B, L)
    non_pad = ~pad_mask
    return (loss_per_tok * non_pad).sum() / non_pad.sum().clamp(min=1)


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Closed-form KL(N(mu,sigma^2) || N(0,I)), per latent dim, batch-mean.
    Returns (d_z,) tensor — used both for the free-bits floor and for the
    per-dim KL histogram logging required by approach/step-1.md §4.
    """
    return 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).mean(dim=0)


def free_bits_kl(kl_dims: torch.Tensor, free_bits: float) -> torch.Tensor:
    """Clamp each dim's KL to a floor of `free_bits` nats before summing —
    the sum is what's actually backpropped (kl_effective).
    """
    return torch.clamp(kl_dims, min=free_bits).sum()


def lipschitz_penalty(
    model: SequenceVAE,
    z: torch.Tensor,
    logits: torch.Tensor,
    sigma_frac: float,
) -> torch.Tensor:
    """Sample delta ~ N(0, sigma^2 I) with sigma proportional to the current
    batch's empirical latent std, decode z+delta, and penalize output-logit
    divergence normalized by ||delta||^2 — a discrete proxy for a Lipschitz
    bound relating latent-space and output-space distance.
    """
    with torch.no_grad():
        sigma = z.std(dim=0, keepdim=True) * sigma_frac + 1e-6
    delta = torch.randn_like(z) * sigma
    logits_pert = model.decode(z + delta, seq_len=logits.shape[1])

    diff = (logits_pert - logits).pow(2).sum(dim=(1, 2))  # (B,)
    delta_norm_sq = delta.pow(2).sum(dim=1).clamp(min=1e-6)  # (B,)
    return (diff / delta_norm_sq).mean()


def lipschitz_penalty_normalized(
    model: SequenceVAE,
    z: torch.Tensor,
    seq_len: int,
    pad_mask: torch.Tensor,
    sigma_frac: float,
) -> torch.Tensor:
    """Scale-free, dropout-free version of the smoothness penalty.

    Perturb each latent dimension by `sigma_frac` of its batch std, decode both z and z + delta
    with the decoder's dropout OFF, and measure how much the per-position token *probabilities*
    change, averaged over real (non-PAD) positions, divided by `sigma_frac`^2. It reads as
    "squared probability change per unit of relative latent perturbation": bounded per position,
    independent of sequence length, logit scale and the absolute perturbation size.

    Why dropout off: the legacy penalty compared a dropout-on decode of z with a separate
    dropout-on decode of z + delta, so the two differed by dropout noise even for delta = 0.
    Measured on trained models, 92-95% of the legacy value was that noise, not sensitivity to z
    (with dropout off the same quantity is ~10x smaller). It also summed raw logit changes over
    every position including padding, giving 10^3-scale values that swamped the reconstruction loss.
    Costs two extra decoder passes while active.
    """
    with torch.no_grad():
        std = z.detach().std(dim=0, keepdim=True) + 1e-6
    delta = torch.randn_like(z) * std * sigma_frac
    was_training = model.decoder.training
    model.decoder.eval()
    try:
        ref = model.decode(z, seq_len=seq_len)
        pert = model.decode(z + delta, seq_len=seq_len)
    finally:
        model.decoder.train(was_training)
    sq = (F.softmax(pert, dim=-1) - F.softmax(ref, dim=-1)).pow(2).sum(dim=-1)  # (B, L)
    keep = ~pad_mask
    return (sq * keep).sum() / keep.sum().clamp(min=1) / sigma_frac**2


def length_prediction_loss(
    model: SequenceVAE, z: torch.Tensor, pad_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (loss, exact-length accuracy).

    Classifier head: cross-entropy over the possible lengths -- exact length is
    what caps the exact-match rate, so it is trained as a classification, not
    a regression that drifts toward the mean length. Regression head (old
    checkpoints): MSE on length / raw_max_length. Trained on z (not mu) to
    match what stage 2 will feed it at generation time.
    """
    true_len = (~pad_mask).sum(dim=1) - 2  # exclude BOS/EOS
    if model.length_head_type == "classifier":
        logits = model.length_logits(z)
        target = (true_len - 1).clamp(0, model.raw_max_length - 1)
        loss = F.cross_entropy(logits, target)
        acc = (logits.argmax(dim=-1) == target).float().mean()
    else:
        pred_frac = model.predict_length_frac(z)
        loss = F.mse_loss(pred_frac, true_len.float() / model.raw_max_length)
        pred_len = (pred_frac * model.raw_max_length).round().clamp(1, model.raw_max_length)
        acc = (pred_len == true_len).float().mean()
    return loss, acc


def vae_loss(
    model: SequenceVAE,
    tokens: torch.Tensor,
    pad_mask: torch.Tensor,
    logits: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    z: torch.Tensor,
    beta: float,
    free_bits: float,
    lipschitz_weight: float,
    lipschitz_sigma_frac: float,
    length_weight: float = 0.0,
    lipschitz_mode: str = "legacy",
) -> dict[str, torch.Tensor]:
    recon = reconstruction_loss(logits, tokens, pad_mask)

    kl_dims = kl_per_dim(mu, logvar)
    kl_raw = kl_dims.sum()
    kl_effective = free_bits_kl(kl_dims, free_bits)

    total = recon + beta * kl_effective

    lip = torch.tensor(0.0, device=tokens.device)
    if lipschitz_weight > 0:
        if lipschitz_mode == "normalized":
            lip = lipschitz_penalty_normalized(model, z, logits.shape[1], pad_mask, lipschitz_sigma_frac)
        else:
            lip = lipschitz_penalty(model, z, logits, lipschitz_sigma_frac)
        total = total + lipschitz_weight * lip

    length_loss, length_acc = length_prediction_loss(model, z, pad_mask)
    total = total + length_weight * length_loss

    return {
        "total": total,
        "recon": recon,
        "kl_raw": kl_raw,
        "kl_effective": kl_effective,
        "kl_per_dim": kl_dims,
        "lipschitz_penalty": lip,
        "length_loss": length_loss,
        "length_acc": length_acc,
        "beta": torch.tensor(beta),
    }


def beta_anneal_schedule(step: int, total_steps: int, anneal_frac: float, beta_target: float) -> float:
    """Linear ramp 0 -> beta_target over the first `anneal_frac` of steps,
    held at beta_target thereafter. Used in phase 1.
    """
    anneal_steps = max(1, int(total_steps * anneal_frac))
    if step >= anneal_steps:
        return beta_target
    return beta_target * (step / anneal_steps)


def beta_rewarm_schedule(
    step: int, total_steps: int, rewarm_frac: float, beta_start: float, beta_target: float
) -> float:
    """Linear ramp beta_start -> beta_target over the first `rewarm_frac` of
    steps, held at beta_target thereafter. Used in phase 2 — starts above 0
    so a KL floor is active from step 1 of fine-tuning (no near-deterministic-
    encoder memorization window on the small target-train set).
    """
    rewarm_steps = max(1, int(total_steps * rewarm_frac))
    if step >= rewarm_steps:
        return beta_target
    frac = step / rewarm_steps
    return beta_start + (beta_target - beta_start) * frac


def lipschitz_active(step: int, total_steps: int, start_frac: float) -> bool:
    return step >= int(total_steps * start_frac)


def lipschitz_weight_at(step: int, total_steps: int, start_frac: float, ramp_frac: float, weight: float) -> float:
    """0 before `start_frac` of training, then ramps linearly to `weight` over `ramp_frac` of
    training (0 = switch on abruptly, the old behaviour), so the penalty doesn't arrive as a shock."""
    start = int(total_steps * start_frac)
    if step < start:
        return 0.0
    ramp = int(total_steps * ramp_frac)
    return weight if ramp <= 0 else weight * min(1.0, (step - start + 1) / ramp)
