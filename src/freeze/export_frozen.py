"""§6 Freeze and handoff to stage 2.

Only run this after eval/freeze_report.py --split test has passed on the
finetune checkpoint you're exporting. This script:
  1. Computes latent_mean/latent_std over target-train mu's and stores them
     as buffers on the model (a pure affine transform, decoder weights
     untouched -- stage 2 consumes normalized latents, un-normalizes before
     decoding).
  2. Saves the frozen checkpoint, versioned distinctly (never overwritten --
     a retrain produces vae_stage1_frozen_v2.pt, since stage 2 is only valid
     paired with the exact frozen checkpoint it trained against).
  3. Encodes ALL 7,433 target sequences (train+val+test -- the held-out split
     only mattered for §5 validation, not for what stage 2 trains on) to the
     handoff parquet.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from src.eval.common import encode_sequences
from src.eval.freeze_report import build_model_from_checkpoint
from src.train.utils import get_device, load_yaml


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finetune-checkpoint", required=True)
    ap.add_argument("--freeze-report", required=True, help="path to the passing test-split freeze_report.json")
    ap.add_argument("--data-config", default="config/data.yaml")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--out-checkpoint", default=None)
    ap.add_argument("--out-latents", default=None)
    args = ap.parse_args()

    report = json.loads(Path(args.freeze_report).read_text())
    if not report.get("overall_pass"):
        raise SystemExit(
            f"Refusing to freeze: {args.freeze_report} does not report overall_pass=true. "
            "Fix the failing gate(s) first -- see diagnostic_pointers in that report."
        )

    device = get_device()
    model = build_model_from_checkpoint(args.finetune_checkpoint, device)

    data_cfg = load_yaml(args.data_config)
    df = pd.read_csv(data_cfg["finetune_clean"])
    train_seqs = df[df["split"] == "train"]["sequence"].tolist()

    train_mu = encode_sequences(model, train_seqs, device, sample=False)
    latent_mean = train_mu.mean(dim=0)
    latent_std = train_mu.std(dim=0).clamp(min=1e-6)
    model.latent_mean.copy_(latent_mean)
    model.latent_std.copy_(latent_std)
    print(f"computed latent_mean/std over {len(train_seqs)} target-train sequences")

    out_ckpt = args.out_checkpoint or f"checkpoints/vae_stage1_frozen_{args.version}.pt"
    if Path(out_ckpt).exists():
        raise SystemExit(f"{out_ckpt} already exists -- bump --version, never overwrite a frozen checkpoint.")

    payload = {
        "model_state_dict": model.state_dict(),
        "model_cfg": torch.load(args.finetune_checkpoint, map_location="cpu", weights_only=False)["model_cfg"],
        "source_finetune_checkpoint": args.finetune_checkpoint,
        "source_finetune_checkpoint_sha256": file_sha256(args.finetune_checkpoint),
        "freeze_report_path": args.freeze_report,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": args.version,
    }
    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_ckpt)
    print(f"wrote frozen checkpoint -> {out_ckpt}")

    # encode ALL target sequences (train+val+test) for stage 2's handoff
    all_df = df[["id", "sequence", "split"]].reset_index(drop=True)
    mu_all = encode_sequences(model, all_df["sequence"].tolist(), device, sample=False)
    mu_norm = model.normalize(mu_all.to(device)).cpu()

    out_latents = args.out_latents or f"data/processed/target_latents_{args.version}.parquet"
    if Path(out_latents).exists():
        raise SystemExit(f"{out_latents} already exists -- bump --version, never overwrite a frozen handoff artifact.")

    latents_df = pd.DataFrame(
        {
            "id": all_df["id"],
            "split": all_df["split"],
            "sequence": all_df["sequence"],
            "mu": mu_all.tolist(),
            "mu_normalized": mu_norm.tolist(),
        }
    )
    Path(out_latents).parent.mkdir(parents=True, exist_ok=True)
    latents_df.to_parquet(out_latents, index=False)
    print(f"wrote {len(latents_df)} target latents -> {out_latents}")


if __name__ == "__main__":
    main()
