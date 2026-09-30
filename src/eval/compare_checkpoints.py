"""Side-by-side comparison of checkpoints on the target-val split.

Built for experiments such as the free-bits sweep: for each checkpoint it reports
the information the latent carries (KL), reconstruction quality, property
preservation, prior-sample realism and the diffusion-relevant gates, then prints a
table and writes it to JSON. Only ever touches val, never test.

    python -m src.eval.compare_checkpoints checkpoints/a.pt checkpoints/b.pt --out reports/compare_x.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.dataset import collate_fn
from src.data.tokenizer import PAD_ID, VOCAB
from src.eval.freeze_report import build_model_from_checkpoint, run_freeze_report
from src.eval.properties import aggregate_posterior_sample_properties, reconstruct
from src.train.utils import get_device, load_yaml


@torch.no_grad()
def val_kl_and_ce(model, seqs, device):
    """Per-dim KL (batch-mean) and token cross-entropy with z = mu, over val."""
    kls, ce_sum, n_tok = [], 0.0, 0
    for i in range(0, len(seqs), 128):
        b = collate_fn([VOCAB.encode(s) for s in seqs[i : i + 128]])
        tok, pm = b["tokens"].to(device), b["pad_mask"].to(device)
        mu, lv = model.encode(tok, pm)
        kls.append((0.5 * (mu**2 + lv.exp() - 1 - lv)).cpu())
        logits = model.decode(mu, seq_len=tok.shape[1])
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tok.reshape(-1), reduction="none")
        keep = (tok != PAD_ID).reshape(-1)
        ce_sum += ce[keep].sum().item()
        n_tok += keep.sum().item()
    return torch.cat(kls).mean(0), ce_sum / n_tok


def evaluate_checkpoint(path, val, train, device):
    model = build_model_from_checkpoint(path, device).eval()
    kl, ce = val_kl_and_ce(model, val, device)
    rec = reconstruct(model, val, device)
    rep = run_freeze_report(model, val, train, device)
    props, prior = rep["properties"], rep["prior_samples"]
    fit = aggregate_posterior_sample_properties(model, train, device)
    pos = float(np.mean([np.mean([a == b for a, b in zip(s, d)]) for s, d in zip(val, rec) if len(s) == len(d)]))
    return {
        "checkpoint": str(path),
        "val_token_ce": ce,
        "val_kl_total": float(kl.sum()),
        "dims_above_0.05": int((kl > 0.05).sum()),
        "max_dim_kl": float(kl.max()),
        "edit_distance": rep["reconstruction"]["mean_edit_distance"],
        "exact_match": rep["reconstruction"]["exact_match_rate"],
        "position_accuracy": pos,
        "length_exact_rate": rep["reconstruction"]["length_exact_rate"],
        "charge_r": props["net_charge"]["pearson_r"],
        "gravy_r": props["gravy"]["pearson_r"],
        "composition_cosine": props["composition_cosine"]["matched"],
        "prior_nondegenerate": rep["density_holes"]["prior_sample_nondegeneracy"]["nondegenerate_rate"],
        "prior_charge_mean": prior["samples"]["net_charge"][0],
        "fit_charge_mean": fit["charge"][0],
        "fit_charge_sd": fit["charge"][1],
        "fit_comp_l1": fit["comp_L1"],
        "fit_degenerate": fit["degenerate_frac"],
        "real_latent_copy_frac": fit["real_latent_copy_frac"],
        "real_charge_mean": prior["real"]["net_charge"][0],
        "mmd2": rep["density_holes"]["aggregate_posterior_vs_prior"]["mmd2_vs_prior"],
        "perturb_spearman": rep["posterior_collapse"]["perturbation_sensitivity"]["spearman_corr"],
        "interp_step": rep["smoothness"]["interpolation_walks"]["mean_consecutive_edit_distance"],
        "gates": rep["gate_results"],
        "prior_examples": prior["examples"],
        "fit_examples": fit["examples"],
    }


COLS = [
    ("val_token_ce", "valCE", "{:.3f}"), ("val_kl_total", "KLnats", "{:.1f}"), ("max_dim_kl", "maxKL/d", "{:.2f}"),
    ("edit_distance", "edit", "{:.3f}"), ("position_accuracy", "posAcc", "{:.3f}"), ("exact_match", "exact", "{:.3f}"),
    ("charge_r", "chg r", "{:.2f}"), ("gravy_r", "gravy r", "{:.2f}"), ("composition_cosine", "compCos", "{:.2f}"),
    ("fit_charge_mean", "fitChg", "{:+.1f}"), ("fit_comp_l1", "fitL1", "{:.3f}"), ("real_latent_copy_frac", "copy%", "{:.1%}"),
    ("prior_nondegenerate", "nondeg", "{:.3f}"),
    ("mmd2", "MMD2", "{:.4f}"), ("perturb_spearman", "pert rho", "{:.2f}"), ("interp_step", "interp", "{:.3f}"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--data-config", default="config/data.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = get_device()
    df = pd.read_csv(load_yaml(args.data_config)["finetune_clean"])
    val = df[df.split == "val"].sequence.tolist()
    train = df[df.split == "train"].sequence.tolist()

    rows = []
    for ck in args.checkpoints:
        print(f"evaluating {ck} ...", flush=True)
        rows.append(evaluate_checkpoint(ck, val, train, device))

    names = [Path(r["checkpoint"]).stem for r in rows]
    w = max(len(n) for n in names)
    print("\n" + "name".ljust(w) + " | " + " | ".join(c[1].rjust(8) for c in COLS) + " | gates F/P/D/S")
    for n, r in zip(names, rows):
        g = r["gates"]
        gs = "/".join("P" if g[k] else "F" for k in ["faithfulness", "posterior_collapse", "density_holes", "smoothness"])
        print(n.ljust(w) + " | " + " | ".join(f.format(r[k]).rjust(8) for k, _, f in COLS) + f" | {gs}")
    print("\nreal-data mean charge:", f"{rows[0]['real_charge_mean']:+.2f}", "(compare with fitChg: samples from a Gaussian fitted to the encoder's real latents, not N(0,I))")
    print("fitL1 = composition distance of those samples to real (held-out real peptides score ~0.18); copy% = decoding real latents returns a near-copy of a training peptide")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
