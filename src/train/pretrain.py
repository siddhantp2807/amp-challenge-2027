"""Phase 1: pretrain on the broad ~28k-sequence corpus.

No held-out split from this corpus is ever used for §5 validation decisions
(those run only against the target-set val/test split in finetune-corpus.csv).
The small dev holdout here exists purely to sanity-check LR/epoch-count
choices during phase-1 training.
"""
import argparse
import math
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.dataset import BucketBatchSampler, PeptideDataset, bucketed_batches, collate_fn
from src.model.vae import SequenceVAE
from src.train.losses import beta_anneal_schedule, lipschitz_weight_at, vae_loss
from src.train.utils import EarlyStopping, get_device, load_yaml, save_checkpoint, set_seed


def build_model(model_cfg: dict) -> SequenceVAE:
    return SequenceVAE(
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        encoder_layers=model_cfg["encoder_layers"],
        decoder_layers=model_cfg["decoder_layers"],
        max_len=model_cfg["max_len"],
        d_z=model_cfg["d_z"],
        decoder_d_model=model_cfg.get("decoder_d_model"),
        decoder_nhead=model_cfg.get("decoder_nhead"),
        decoder_dim_feedforward=model_cfg.get("decoder_dim_feedforward"),
        length_head=model_cfg.get("length_head", "regression"),
        length_hidden=model_cfg.get("length_hidden", 256),
    )


def lr_lambda(step: int, total_steps: int, warmup_frac: float):
    warmup_steps = max(1, int(total_steps * warmup_frac))
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    totals = {"recon": 0.0, "kl_raw": 0.0, "kl_effective": 0.0, "length_acc": 0.0}
    n = 0
    for batch in loader:
        tokens = batch["tokens"].to(device)
        pad_mask = batch["pad_mask"].to(device)
        mu, logvar = model.encode(tokens, pad_mask)
        z = model.reparameterize(mu, logvar)
        logits = model.decode(z, seq_len=tokens.shape[1])
        out = vae_loss(
            model, tokens, pad_mask, logits, mu, logvar, z,
            beta=1.0, free_bits=0.0, lipschitz_weight=0.0, lipschitz_sigma_frac=0.0,
        )
        totals["recon"] += out["recon"].item()
        totals["kl_raw"] += out["kl_raw"].item()
        totals["kl_effective"] += out["kl_effective"].item()
        totals["length_acc"] += out["length_acc"].item()
        n += 1
    model.train()
    return {k: v / max(1, n) for k, v in totals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", default="config/data.yaml")
    ap.add_argument("--model-config", default="config/model.yaml")
    ap.add_argument("--train-config", default="config/train_pretrain.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.train_config)

    df = pd.read_csv(data_cfg["pretrain_clean"])
    sequences = df["sequence"].tolist()

    rng = torch.Generator().manual_seed(args.seed)
    n_dev = max(1, int(len(sequences) * data_cfg["pretrain_dev_holdout_frac"]))
    perm = torch.randperm(len(sequences), generator=rng).tolist()
    dev_idx = set(perm[:n_dev])
    train_seqs = [s for i, s in enumerate(sequences) if i not in dev_idx]
    dev_seqs = [s for i, s in enumerate(sequences) if i in dev_idx]
    print(f"pretrain: {len(train_seqs)} train, {len(dev_seqs)} dev (monitoring only)")

    train_ds = PeptideDataset(train_seqs)
    dev_ds = PeptideDataset(dev_seqs)

    train_sampler = BucketBatchSampler(
        train_seqs, batch_size=train_cfg["batch_size"],
        bucket_edges=data_cfg["length_buckets"], shuffle=True, seed=args.seed,
    )
    dev_loader = DataLoader(dev_ds, batch_size=train_cfg["batch_size"], shuffle=False, collate_fn=collate_fn)

    model = build_model(model_cfg).to(device)
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
    early_stop = EarlyStopping(patience=train_cfg["patience_epochs"], mode="min")

    step = 0
    best_dev_recon = float("inf")
    best_path = f"{train_cfg['checkpoint_dir']}/{train_cfg['run_name']}_best.pt"
    final_path = f"{train_cfg['checkpoint_dir']}/{train_cfg['run_name']}_final.pt"

    t0 = time.time()
    for epoch in range(train_cfg["max_epochs"]):
        train_sampler.set_epoch(epoch)
        pbar = tqdm(bucketed_batches(train_ds, train_sampler), total=len(train_sampler), desc=f"epoch {epoch}", leave=False)
        for batch in pbar:
            tokens = batch["tokens"].to(device)
            pad_mask = batch["pad_mask"].to(device)

            beta = beta_anneal_schedule(step, total_steps, train_cfg["kl_anneal_frac"], train_cfg["beta_target"])
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
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                writer.add_histogram("train/kl_per_dim", out["kl_per_dim"].detach().cpu(), step)
                pbar.set_postfix(
                    recon=f"{out['recon'].item():.3f}",
                    kl=f"{out['kl_raw'].item():.3f}",
                    beta=f"{beta:.3f}",
                    len_loss=f"{out['length_loss'].item():.3f}",
                    len_acc=f"{out['length_acc'].item():.2f}",
                )

            step += 1

        dev_metrics = evaluate(model, dev_loader, device)
        elapsed = time.time() - t0
        tqdm.write(
            f"epoch {epoch} step {step} dev_recon {dev_metrics['recon']:.4f} "
            f"dev_kl_raw {dev_metrics['kl_raw']:.4f} dev_len_acc {dev_metrics['length_acc']:.3f} elapsed {elapsed:.0f}s"
        )
        writer.add_scalar("dev/recon", dev_metrics["recon"], step)
        writer.add_scalar("dev/kl_raw", dev_metrics["kl_raw"], step)
        writer.add_scalar("dev/length_acc", dev_metrics["length_acc"], step)

        if dev_metrics["recon"] < best_dev_recon:
            best_dev_recon = dev_metrics["recon"]
            save_checkpoint(best_path, model, optimizer, extra={"epoch": epoch, "step": step, "model_cfg": model_cfg})

        if early_stop.step(dev_metrics["recon"]):
            tqdm.write(f"early stopping at epoch {epoch}")
            break

    save_checkpoint(final_path, model, optimizer, extra={"epoch": epoch, "step": step, "model_cfg": model_cfg})
    print(f"saved best -> {best_path}, final -> {final_path}")


if __name__ == "__main__":
    main()
