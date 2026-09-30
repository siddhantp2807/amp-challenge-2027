"""Stage-2 training loop. Mirrors src/train/finetune.py's conventions.

Two phases, selected by `phase` in the train config:
  union -> the ~31.1k deduplicated broad+target corpus (phase A)
  train -> target-train only, initialized from phase A (phase B)

Validation is target-val in both cases. The VAE is loaded read-only and is never
frozen or modified; stage 2 owns its own latent standardization (see latents.py).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.diffusion.conditioning import PROPERTIES, PropertyStandardizer
from src.diffusion.data import (build_split_frames, encode_frame, sample_latents,
                                standardized_conditions)
from src.diffusion.denoiser import build_denoiser, n_params
from src.diffusion.diffusion import EMA, LatentDiffusion
from src.diffusion.latents import LatentStats, load_vae, posterior_width_report
from src.diffusion.schedule import NoiseSchedule
from src.train.pretrain import lr_lambda
from src.train.utils import (EarlyStopping, get_device, load_yaml, save_checkpoint,
                             set_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default="config/data.yaml")
    ap.add_argument("--model-config", default="config/diffusion_model.yaml")
    ap.add_argument("--train-config", default="config/train_diffusion_pretrain.yaml")
    ap.add_argument("--vae-checkpoint", default=None,
                    help="override the train config's vae_checkpoint; the effective "
                         "path is what gets recorded in the checkpoint payload")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)
    # applied before anything reads it, so the payload written by extra() records
    # the VAE actually used rather than the config's default
    if args.vae_checkpoint:
        train_cfg["vae_checkpoint"] = args.vae_checkpoint
    run = train_cfg["run_name"]
    print(f"run={run} device={device} phase={train_cfg['phase']}")

    vae = load_vae(train_cfg["vae_checkpoint"], device)
    if vae.d_z != model_cfg["d_z"]:
        raise SystemExit(
            f"d_z mismatch: VAE has {vae.d_z}, diffusion config says {model_cfg['d_z']}"
        )

    frames = build_split_frames(data_cfg)
    train_set = encode_frame(vae, frames[train_cfg["phase"]], device)
    val_set = encode_frame(vae, frames["val"], device)
    print(f"train={len(train_set)} val={len(val_set)}")

    # standardizers are fitted on THIS phase's training data and travel with the
    # checkpoint, so sampling standardizes user targets exactly as training did
    prop_std = PropertyStandardizer.fit(train_set.conditions)
    z_stats = LatentStats.fit(train_set.mu)
    width = posterior_width_report(train_set.logvar, train_set.mu.std(0))
    print(f"G6 posterior width ratio {width['ratio']:.3f} "
          f"({'real' if width['augmentation_is_meaningful'] else 'COSMETIC'} augmentation)")
    if not width["augmentation_is_meaningful"]:
        print("  -> z-resampling is near-cosmetic; consider an explicit jitter sweep")

    c_train = standardized_conditions(train_set, prop_std).to(device)
    c_val = standardized_conditions(val_set, prop_std).to(device)
    it_train = train_set.is_target.to(device)
    it_val = val_set.is_target.to(device)
    z_val = z_stats.normalize(val_set.mu).to(device)

    li = list(PROPERTIES).index("length")
    model = build_denoiser(model_cfg, prop_std.mean[li], prop_std.std[li]).to(device)
    print(f"denoiser: {n_params(model)/1e6:.2f}M params "
          f"({n_params(model)/len(train_set):.0f} per training example)")

    if train_cfg.get("init_from"):
        payload = torch.load(train_cfg["init_from"], map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        print(f"initialized from {train_cfg['init_from']}")

    schedule = NoiseSchedule(model_cfg["timesteps"], model_cfg["schedule_s"], device)
    diff = LatentDiffusion(model, schedule, model_cfg["parameterization"],
                           model_cfg.get("self_cond", False))
    ema = EMA(model, train_cfg["ema_decay"])

    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"],
                            weight_decay=train_cfg["weight_decay"],
                            betas=tuple(train_cfg["betas"]))
    bs = train_cfg["batch_size"]
    steps_per_epoch = max(1, math.ceil(len(train_set) / bs))
    total_steps = steps_per_epoch * train_cfg["max_epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: lr_lambda(s, total_steps, train_cfg["warmup_frac"])
    )

    writer = SummaryWriter(log_dir=f"runs/{run}")
    stopper = EarlyStopping(patience=train_cfg["patience_epochs"], mode="min")
    ckpt_dir = Path(train_cfg["checkpoint_dir"])
    best_path, final_path = ckpt_dir / f"{run}_best.pt", ckpt_dir / f"{run}_final.pt"

    def extra(epoch, step, stop_reason=None):
        return {
            "epoch": epoch, "step": step,
            "model_cfg": model_cfg,
            "vae_checkpoint": train_cfg["vae_checkpoint"],
            "phase": train_cfg["phase"],
            "z_stats": z_stats.state_dict(),
            "prop_stats": prop_std.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "posterior_width": width,
            **({"stop_reason": stop_reason} if stop_reason else {}),
        }

    step, best_val, stop_reason = 0, float("inf"), "max_epochs"
    for epoch in range(train_cfg["max_epochs"]):
        model.train()
        order = torch.randperm(len(train_set))
        running = 0.0
        bar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch}", leave=False)
        for bi in bar:
            idx = order[bi * bs : (bi + 1) * bs]
            z0 = sample_latents(train_set, z_stats, idx,
                                resample=train_cfg["resample_z"],
                                jitter_sigma=train_cfg.get("jitter_sigma", 0.0)).to(device)
            loss = diff.loss(z0, c_train[idx], it_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            opt.step()
            sched.step()
            ema.update(model)
            step += 1
            running += loss.item()
            if step % train_cfg["log_every"] == 0:
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
            bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / steps_per_epoch
        # evaluate the EMA weights, on mu (not resampled z), with fixed noise
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        model.eval()
        val_loss = diff.eval_loss(z_val, c_val, it_val)
        train_eval = diff.eval_loss(
            z_stats.normalize(train_set.mu[:len(val_set)]).to(device),
            c_train[:len(val_set)], it_train[:len(val_set)])
        model.load_state_dict(backup)

        writer.add_scalar("val/loss", val_loss, step)
        writer.add_scalar("train/eval_loss", train_eval, step)
        writer.add_scalar("gap/train_minus_val", train_eval - val_loss, step)
        print(f"epoch {epoch}: train {train_loss:.4f} | val {val_loss:.4f} "
              f"| gap {train_eval - val_loss:+.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(str(best_path), model, opt, extra=extra(epoch, step))
        if stopper.step(val_loss):
            stop_reason = f"val loss plateaued at epoch {epoch}"
            print(stop_reason)
            break

    save_checkpoint(str(final_path), model, opt, extra=extra(epoch, step, stop_reason))
    writer.close()
    print(f"best val {best_val:.4f} -> {best_path}")


if __name__ == "__main__":
    main()
