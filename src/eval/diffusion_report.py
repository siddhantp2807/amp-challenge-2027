"""Stage-2 evaluation. Mirrors freeze_report.py's shape (per-check dicts + verdicts).

Every property-MAE number is printed against two reference rows, because it is
uninterpretable alone:

  decoder ceiling  -- decode the TRUE val latents. Nothing using this decoder can
                      do better. Measured 0.000 / 0.549 / 0.451.
  primary bar      -- the conditional GMM, seed-averaged. Not the linear-Gaussian:
                      p(z|c) is multimodal in length, so a single ellipsoid per
                      condition under-fits and would be too soft a target.

Both are recomputed here rather than hardcoded, so the comparison always reflects
the same VAE checkpoint and split the diffusion model was scored on.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.diffusion.baseline import ConditionalGaussian, ConditionalGMM
from src.diffusion.conditioning import PROPERTIES
from src.diffusion.data import build_split_frames, encode_frame
from src.diffusion.latents import load_vae
from src.diffusion.metrics import decode_latents, mmd2, property_mae, score_latents
from src.diffusion.sample import load_diffusion
from src.eval.common import decode_with_predicted_length
from src.train.utils import get_device, load_yaml, set_seed


def forced_length_decode(vae, z, device, lengths, batch_size: int = 128):
    """Decode but truncate to a requested length instead of the predicted one.

    Separates 'the diffusion model put z in the wrong place' from 'the length head
    read z differently than intended'. With length_exact_rate 1.0 on this VAE the
    two should agree closely; a divergence localizes the fault.
    """
    from src.data.tokenizer import SPECIAL_TOKENS, VOCAB
    n_special = len(SPECIAL_TOKENS)
    out = []
    with torch.no_grad():
        for i in range(0, len(z), batch_size):
            chunk = z[i : i + batch_size].to(device)
            logits = vae.decode(chunk, seq_len=vae.max_len)
            aa = logits[:, 1:, n_special:].argmax(-1).cpu() + n_special
            for j, row in enumerate(aa):
                n = int(lengths[i + j])
                out.append(VOCAB.decode(row[:n].tolist(), stop_at_eos=False))
    return out


@torch.no_grad()
def terminal_latent_error(diff, z_stats, z_true, c, is_target, mask, device,
                          t_frac: float = 0.5, steps: int = 50, seed: int = 0) -> dict:
    """Noise real latents to t, denoise back conditioned on their own properties.

    This is the measurement SUMMARY.md asks for: a *measured* radius to replace
    the hardcoded 0.25 sigma in the VAE's noise_survival check, and to define
    smoothness over the neighbourhood the sampler actually visits rather than
    along paths between random peptide pairs.
    """
    g = torch.Generator().manual_seed(seed)
    zt_index = int(t_frac * (diff.schedule.timesteps - 1))
    z0 = z_stats.normalize(z_true).to(device)
    noise = torch.randn(z0.shape, generator=g).to(device)
    t = torch.full((len(z0),), zt_index, device=device, dtype=torch.long)
    x_t = diff.schedule.q_sample(z0, t, noise)

    # resume the reverse process from t rather than from pure noise
    ts = torch.linspace(zt_index, 0, steps).long().to(device)
    x = x_t
    for i, tv in enumerate(ts):
        tt = tv.expand(len(x))
        pred = diff.model(x, tt, c, mask, is_target)
        x0 = diff.schedule.to_x0(pred, x, tt, diff.parameterization)
        if i == len(ts) - 1:
            x = x0
            break
        ab_next = diff.schedule.alpha_bar[ts[i + 1]].unsqueeze(-1)
        eps = diff.schedule.to_eps(x0, x, tt)
        x = ab_next.sqrt() * x0 + (1 - ab_next).clamp(min=0).sqrt() * eps

    err = (x - z0).norm(dim=-1)
    per_dim = (x - z0).std(0).mean()
    return {
        "t_frac": t_frac,
        "mean_l2": float(err.mean()),
        "per_dim_sigma": float(per_dim),
        "note": "empirically-derived replacement for noise_survival's hardcoded 0.25",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-config", default="config/data.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg-weights", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--n-per-condition", type=int, default=8)
    ap.add_argument("--gmm-components", type=int, default=16)
    ap.add_argument("--baseline-seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--vae-checkpoint", default=None,
                    help="override the VAE path baked into the diffusion checkpoint")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    diff, prop_std, z_stats, payload = load_diffusion(args.checkpoint, device)
    vae_checkpoint = args.vae_checkpoint or payload["vae_checkpoint"]
    vae = load_vae(vae_checkpoint, device)

    frames = build_split_frames(load_yaml(args.data_config))
    tr = encode_frame(vae, frames["train"], device)
    ev = encode_frame(vae, frames[args.split], device)
    c_raw, train_std = ev.conditions, tr.conditions.std(0)
    n = len(ev)

    c = torch.tensor(prop_std.transform(c_raw), dtype=torch.float32, device=device)
    mask = torch.ones(n, len(PROPERTIES), dtype=torch.bool, device=device)
    is_target = torch.ones(n, dtype=torch.long, device=device)

    report = {
        "checkpoint": args.checkpoint,
        "vae_checkpoint": vae_checkpoint,
        "split": args.split,
        "n": n,
        "properties": list(PROPERTIES),
        "train_std": dict(zip(PROPERTIES, train_std.round(4).tolist())),
        "steps": args.steps,
    }

    # --- reference rows -----------------------------------------------------
    print("decoder ceiling ...")
    report["decoder_ceiling"] = score_latents(
        vae, ev.mu, c_raw, train_std, tr.sequences, device)

    print(f"primary bar: conditional GMM K={args.gmm_components}, "
          f"{args.baseline_seeds} seeds ...")
    gmm = ConditionalGMM.fit(tr.mu.numpy(), tr.conditions, args.gmm_components, args.seed)
    cg = ConditionalGaussian.fit(tr.mu.numpy(), tr.conditions)
    bars = {}
    for name, sampler in (("conditional_gmm", gmm), ("conditional_gaussian", cg)):
        runs = []
        for s in range(args.baseline_seeds):
            z = sampler.sample(c_raw, np.random.default_rng(1000 + s))
            m = score_latents(vae, z, c_raw, train_std, tr.sequences, device)["property_mae"]
            runs.append([m[p] for p in PROPERTIES])
        a = np.array(runs)
        bars[name] = {"mean": dict(zip(PROPERTIES, a.mean(0).round(4).tolist())),
                      "sd": dict(zip(PROPERTIES, a.std(0).round(4).tolist()))}
    report["baselines"] = bars
    report["primary_bar"] = bars["conditional_gmm"]

    # --- the diffusion model, swept over guidance strength ------------------
    print("diffusion samples ...")
    report["cfg_sweep"] = {}
    for w in args.cfg_weights:
        g = torch.Generator().manual_seed(args.seed)
        z_norm = diff.ddim_sample(c, mask, is_target, steps=args.steps,
                                  cfg_weight=w, generator=g)
        z = z_stats.denormalize(z_norm.cpu())
        row = score_latents(vae, z, c_raw, train_std, tr.sequences, device)
        row["mmd2_vs_real_latents"] = mmd2(z.numpy(), ev.mu.numpy())
        row["forced_length_mae"] = property_mae(
            forced_length_decode(vae, z, device, c_raw[:, list(PROPERTIES).index("length")]),
            c_raw, train_std)
        report["cfg_sweep"][f"w={w}"] = row
        m = row["property_mae"]
        print(f"  w={w:<4} " + "  ".join(f"{p}={m[p]:.3f}" for p in PROPERTIES)
              + f"  uniq={row['unique_frac']:.3f} copy={row.get('copy_frac', float('nan')):.3f}"
              + f" degen={row['degenerate_frac']:.3f}")

    best_w = min(report["cfg_sweep"],
                 key=lambda k: np.mean([report["cfg_sweep"][k]["property_mae"][p]
                                        for p in PROPERTIES]))
    report["best_cfg_weight"] = best_w
    best = report["cfg_sweep"][best_w]["property_mae"]

    # --- diversity under a FIXED condition ----------------------------------
    print("diversity under fixed condition ...")
    k = min(32, n)
    rep_c = c[:k].repeat_interleave(args.n_per_condition, 0)
    rep_mask = mask[:k].repeat_interleave(args.n_per_condition, 0)
    g = torch.Generator().manual_seed(args.seed + 1)
    zz = z_stats.denormalize(diff.ddim_sample(
        rep_c, rep_mask, torch.ones(len(rep_c), dtype=torch.long, device=device),
        steps=args.steps, cfg_weight=float(best_w.split("=")[1]), generator=g).cpu())
    seqs = decode_latents(vae, zz, device)
    from src.eval.common import normalized_edit_distance
    per_cond = []
    for i in range(k):
        grp = seqs[i * args.n_per_condition : (i + 1) * args.n_per_condition]
        d = [normalized_edit_distance(a, b)
             for ai, a in enumerate(grp) for b in grp[ai + 1:]]
        per_cond.append(float(np.mean(d)) if d else 0.0)
    report["fixed_condition_diversity"] = {
        "n_conditions": k, "n_per_condition": args.n_per_condition,
        "mean_pairwise_edit_distance": float(np.mean(per_cond)),
        "unique_frac": len(set(seqs)) / len(seqs),
    }

    print("terminal latent error ...")
    report["terminal_latent_error"] = terminal_latent_error(
        diff, z_stats, ev.mu, c, is_target, mask, device, steps=args.steps, seed=args.seed)

    # --- verdicts -----------------------------------------------------------
    bar, ceil = bars["conditional_gmm"], report["decoder_ceiling"]["property_mae"]
    beats = {p: bool(best[p] < bar["mean"][p]) for p in PROPERTIES}
    report["verdict"] = {
        "beats_primary_bar": beats,
        "beats_primary_bar_all": all(beats.values()),
        "near_ceiling": {p: bool(best[p] - ceil[p] < 0.05) for p in PROPERTIES},
        "note": (
            "A property already within ~0.05 std of the decoder ceiling is solved; "
            "failing to beat the bar there is not a modelling failure. Check "
            "near_ceiling before reading beats_primary_bar as a verdict."
        ),
    }

    out = args.out or f"reports/diffusion_report_{args.split}_{Path(args.checkpoint).stem}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, default=float))

    print(f"\n{'property':10s} {'diffusion':>10s} {'GMM bar':>10s} {'ceiling':>10s}  verdict")
    for p in PROPERTIES:
        flag = "SOLVED" if report["verdict"]["near_ceiling"][p] else (
            "beats bar" if beats[p] else "below bar")
        print(f"{p:10s} {best[p]:10.3f} {bar['mean'][p]:10.3f} {ceil[p]:10.3f}  {flag}")
    print(f"\nbest cfg {best_w} -> {out}")


if __name__ == "__main__":
    main()
