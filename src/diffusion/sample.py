"""Sample peptides from a trained stage-2 model. CLI mirrors src/generate.py.

Any subset of the properties can be specified; whatever is left out is masked,
which is exactly the configuration the model was trained on (k ~ U{0..3} keeps a
random subset each step). Specifying nothing gives unconditional generation.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from src.diffusion.conditioning import PROPERTIES, PropertyStandardizer
from src.diffusion.denoiser import build_denoiser
from src.diffusion.diffusion import LatentDiffusion
from src.diffusion.latents import LatentStats, load_vae
from src.diffusion.metrics import decode_latents
from src.diffusion.schedule import NoiseSchedule
from src.train.utils import get_device, set_seed


def load_diffusion(checkpoint: str, device, use_ema: bool = True):
    """Rebuild the denoiser from the checkpoint's own config, like the VAE's
    build_model_from_checkpoint does. Defaults to the EMA weights, which are what
    training selected the best epoch on."""
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = payload["model_cfg"]
    prop_std = PropertyStandardizer.from_state_dict(payload["prop_stats"])
    z_stats = LatentStats.from_state_dict(payload["z_stats"])
    li = list(PROPERTIES).index("length")
    model = build_denoiser(cfg, prop_std.mean[li], prop_std.std[li]).to(device)
    state = payload.get("ema_state_dict") if use_ema else None
    model.load_state_dict(state or payload["model_state_dict"])
    model.eval()
    schedule = NoiseSchedule(cfg["timesteps"], cfg["schedule_s"], device)
    diff = LatentDiffusion(model, schedule, cfg["parameterization"],
                           cfg.get("self_cond", False))
    return diff, prop_std, z_stats, payload


def build_conditions(n: int, requested: dict, prop_std: PropertyStandardizer,
                     device) -> tuple[torch.Tensor, torch.Tensor]:
    """(standardized values, active mask). Unspecified properties are masked.

    Masked columns are filled with the training mean so the standardized value is
    0 -- the embedder ignores them anyway (it substitutes the learned mask token),
    but a NaN or a wild value there would be a silent hazard if that ever changed.
    """
    raw = np.tile(prop_std.mean, (n, 1))
    mask = np.zeros((n, len(PROPERTIES)), dtype=bool)
    for i, p in enumerate(PROPERTIES):
        if requested.get(p) is not None:
            raw[:, i] = requested[p]
            mask[:, i] = True
    c = torch.tensor(prop_std.transform(raw), dtype=torch.float32, device=device)
    return c, torch.tensor(mask, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vae-checkpoint", default=None,
                    help="defaults to the VAE the diffusion checkpoint was trained against")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg-weight", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--cads", action="store_true", help="condition-annealed sampling")
    ap.add_argument("--cads-mode", default="mask", choices=["mask", "value"])
    ap.add_argument("--no-ema", action="store_true")
    for p in PROPERTIES:
        ap.add_argument(f"--{p}", type=float, default=None,
                        help=f"target {p}; omit to leave it unconstrained")
    ap.add_argument("--out-prefix", default=None,
                    help="writes <prefix>.fasta and <prefix>.csv")
    ap.add_argument("--out-fasta", default=None,
                    help="exact FASTA path; a sibling .csv with the measured "
                         "properties is written alongside it")
    args = ap.parse_args()
    if args.out_prefix and args.out_fasta:
        ap.error("pass --out-prefix or --out-fasta, not both")

    set_seed(args.seed)
    device = get_device()
    diff, prop_std, z_stats, payload = load_diffusion(
        args.checkpoint, device, use_ema=not args.no_ema)
    vae = load_vae(args.vae_checkpoint or payload["vae_checkpoint"], device)

    requested = {p: getattr(args, p) for p in PROPERTIES}
    named = {k: v for k, v in requested.items() if v is not None}
    print(f"conditioning on {named or 'nothing (unconditional)'}")
    c, mask = build_conditions(args.n, requested, prop_std, device)
    is_target = torch.ones(args.n, dtype=torch.long, device=device)

    g = torch.Generator().manual_seed(args.seed)
    z_norm = diff.ddim_sample(
        c, mask, is_target, steps=args.steps, cfg_weight=args.cfg_weight, eta=args.eta,
        cads={"mode": args.cads_mode} if args.cads else None, generator=g,
    )
    z = z_stats.denormalize(z_norm.cpu())
    seqs = decode_latents(vae, z, device)

    from src.diffusion.metrics import compute_properties
    props = compute_properties(seqs)
    print(f"\n{len(seqs)} samples, {len(set(seqs))} unique")
    for i, p in enumerate(PROPERTIES):
        line = f"  {p:8s} mean {props[:, i].mean():7.3f}  sd {props[:, i].std():6.3f}"
        if requested[p] is not None:
            line += f"   target {requested[p]:7.3f}   MAE {np.abs(props[:, i] - requested[p]).mean():.3f}"
        print(line)
    print("\n".join(f"  {s}" for s in seqs[:10]))

    if args.out_prefix or args.out_fasta:
        fasta = Path(args.out_fasta) if args.out_fasta else Path(f"{args.out_prefix}.fasta")
        csv_path = fasta.with_suffix(".csv")
        fasta.parent.mkdir(parents=True, exist_ok=True)

        # the FASTA header carries everything needed to regenerate this exact file
        cond = ",".join(f"{p}={requested[p]:g}" for p in PROPERTIES
                        if requested[p] is not None) or "unconditional"
        stamp = (f"seed={args.seed} steps={args.steps} cfg={args.cfg_weight} "
                 f"{cond} ckpt={Path(args.checkpoint).name}")
        with open(fasta, "w") as f:
            f.write(f"; {stamp}\n")
            for i, s in enumerate(seqs):
                f.write(f">gen_{i} {stamp}\n{s}\n")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "sequence", *PROPERTIES])
            for i, s in enumerate(seqs):
                w.writerow([f"gen_{i}", s, *props[i].round(4)])
        print(f"wrote {fasta} and {csv_path}")


if __name__ == "__main__":
    main()
