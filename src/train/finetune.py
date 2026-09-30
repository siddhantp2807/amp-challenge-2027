"""Phase 2: continue training (never freeze) on the target-train split.

Smoothness-specific choices vs. phase 1, all aimed at preventing the small
(6,690-sequence) fine-tune set from carving sharp local basins/discontinuities
into the pretrained latent geometry: much lower LR, a KL *re-warm* (not a
re-anneal from 0), a higher Lipschitz-penalty weight, and an early-stopping OR
condition on recon-on-val plateau OR the interpolation-walk smoothness proxy
degrading.
"""
import argparse
import time

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.dataset import BucketBatchSampler, PeptideDataset, bucketed_batches
from src.eval.reconstruction import reconstruction_fidelity
from src.eval.smoothness import interpolation_monotonicity_proxy
from src.train.losses import beta_rewarm_schedule, lipschitz_weight_at, vae_loss
from src.train.pretrain import build_model, lr_lambda
from src.train.utils import EarlyStopping, get_device, load_checkpoint, load_yaml, save_checkpoint, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default="config/data.yaml")
    ap.add_argument("--model-config", default="config/model.yaml")
    ap.add_argument("--train-config", default="config/train_finetune.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)

    df = pd.read_csv(data_cfg["finetune_clean"])
    train_seqs = df[df["split"] == "train"]["sequence"].tolist()
    val_seqs = df[df["split"] == "val"]["sequence"].tolist()
    print(f"finetune: {len(train_seqs)} train, {len(val_seqs)} val (eval only)")

    train_ds = PeptideDataset(train_seqs)
    train_sampler = BucketBatchSampler(
        train_seqs, batch_size=train_cfg["batch_size"],
        bucket_edges=data_cfg["length_buckets"], shuffle=True, seed=args.seed,
    )
    model = build_model(model_cfg).to(device)
    load_checkpoint(train_cfg["init_from"], model, map_location=device)
    print(f"initialized from {train_cfg['init_from']}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"],
        betas=tuple(train_cfg["betas"]),
    )

    steps_per_epoch = len(train_sampler)
    total_steps = steps_per_epoch * train_cfg["max_epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: lr_lambda(s, total_steps, train_cfg["warmup_frac"])
    )

    writer = SummaryWriter(log_dir=f"runs/{train_cfg['run_name']}")
    recon_early_stop = EarlyStopping(patience=train_cfg["patience_epochs"], mode="min")
    smoothness_early_stop = EarlyStopping(patience=train_cfg["smoothness_patience_epochs"], mode="min")

    step = 0
    best_val_recon = float("inf")
    best_path = f"{train_cfg['checkpoint_dir']}/{train_cfg['run_name']}_best.pt"
    final_path = f"{train_cfg['checkpoint_dir']}/{train_cfg['run_name']}_final.pt"

    t0 = time.time()
    stop_reason = None
    for epoch in range(train_cfg["max_epochs"]):
        train_sampler.set_epoch(epoch)
        pbar = tqdm(bucketed_batches(train_ds, train_sampler), total=len(train_sampler), desc=f"epoch {epoch}", leave=False)
        for batch in pbar:
            tokens = batch["tokens"].to(device)
            pad_mask = batch["pad_mask"].to(device)

            beta = beta_rewarm_schedule(
                step, total_steps, train_cfg["kl_rewarm_frac"],
                train_cfg["beta_target"] * train_cfg["beta_start_frac"], train_cfg["beta_target"],
            )
            lip_w = lipschitz_weight_at(
                step, total_steps, train_cfg["lipschitz_start_frac"],
                train_cfg.get("lipschitz_ramp_frac", 0.0), train_cfg["lipschitz_weight"],
            )

            mu, logvar = model.encode(tokens, pad_mask)
            z = model.reparameterize(mu, logvar)
            logits = model.decode(z, seq_len=tokens.shape[1])
            out = vae_loss(
                model, tokens, pad_mask, logits, mu, logvar, z,
                beta=beta, free_bits=train_cfg["free_bits"],
                lipschitz_weight=lip_w, lipschitz_sigma_frac=train_cfg["lipschitz_sigma_frac"],
                length_weight=train_cfg["length_weight"],
                lipschitz_mode=train_cfg.get("lipschitz_mode", "legacy"),
            )

            optimizer.zero_grad()
            out["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % train_cfg["log_every"] == 0:
                writer.add_scalar("train/recon", out["recon"].item(), step)
                writer.add_scalar("train/kl_raw", out["kl_raw"].item(), step)
                writer.add_scalar("train/kl_effective", out["kl_effective"].item(), step)
                writer.add_scalar("train/lipschitz_penalty", out["lipschitz_penalty"].item(), step)
                writer.add_scalar("train/length_loss", out["length_loss"].item(), step)
                writer.add_scalar("train/length_acc", out["length_acc"].item(), step)
                writer.add_scalar("train/total", out["total"].item(), step)
                writer.add_scalar("train/beta", beta, step)
                writer.add_histogram("train/kl_per_dim", out["kl_per_dim"].detach().cpu(), step)
                pbar.set_postfix(
                    recon=f"{out['recon'].item():.3f}",
                    kl=f"{out['kl_raw'].item():.3f}",
                    beta=f"{beta:.3f}",
                    len_loss=f"{out['length_loss'].item():.3f}",
                    len_acc=f"{out['length_acc'].item():.2f}",
                )

            step += 1

        val_recon_metrics = reconstruction_fidelity(model, val_seqs, device)
        val_recon = val_recon_metrics["mean_edit_distance"]
        smoothness_proxy = interpolation_monotonicity_proxy(
            model, val_seqs, device, n_pairs=train_cfg["smoothness_n_pairs"]
        )
        elapsed = time.time() - t0
        tqdm.write(
            f"epoch {epoch} step {step} val_exact_match {val_recon_metrics['exact_match_rate']:.4f} "
            f"val_edit_dist {val_recon:.4f} val_len_exact {val_recon_metrics['length_exact_rate']:.3f} "
            f"smoothness_proxy {smoothness_proxy:.4f} elapsed {elapsed:.0f}s"
        )
        writer.add_scalar("val/exact_match_rate", val_recon_metrics["exact_match_rate"], step)
        writer.add_scalar("val/mean_edit_distance", val_recon, step)
        writer.add_scalar("val/length_exact_rate", val_recon_metrics["length_exact_rate"], step)
        writer.add_scalar("val/smoothness_proxy", smoothness_proxy, step)

        if val_recon < best_val_recon:
            best_val_recon = val_recon
            save_checkpoint(best_path, model, optimizer, extra={"epoch": epoch, "step": step, "model_cfg": model_cfg})

        recon_stop = recon_early_stop.step(val_recon)
        smoothness_stop = smoothness_early_stop.step(smoothness_proxy)
        if recon_stop or smoothness_stop:
            stop_reason = "recon plateau" if recon_stop else "smoothness proxy degrading"
            tqdm.write(f"early stopping at epoch {epoch}: {stop_reason}")
            break

    save_checkpoint(
        final_path, model, optimizer,
        extra={"epoch": epoch, "step": step, "model_cfg": model_cfg, "stop_reason": stop_reason},
    )
    print(f"saved best -> {best_path}, final -> {final_path}")


if __name__ == "__main__":
    main()
