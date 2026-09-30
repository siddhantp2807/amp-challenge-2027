"""§5.5 Freeze gate: aggregates all §5 checks + extra smoothness diagnostics
into one pass/fail report, with targeted diagnostic pointers on failure
rather than raw numbers only (approach/step-1.md's own language is explicit
that a failing check should redirect effort, not be silently ignored).

Run with --split val during iteration/model-selection; run once with
--split test as the actual freeze decision.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.eval.density_holes import density_holes_check
from src.eval.faithfulness import faithfulness_check
from src.eval.posterior_collapse import posterior_collapse_check
from src.eval.properties import prior_sample_properties, property_preservation
from src.eval.reconstruction import reconstruction_fidelity
from src.eval.smoothness import smoothness_check
from src.model.vae import SequenceVAE
from src.train.utils import get_device, load_checkpoint, load_yaml

# Exact-match reconstruction is no longer gated -- see src/eval/faithfulness.py
# for why (it conflates encoder information loss with decoder error, and is
# unreachable for a non-autoregressive decoder). Reconstruction is still
# reported for diagnostics; the length-head rate below is still worth flagging,
# since it caps exact match and was a real bug twice.
LENGTH_EXACT_RATE_POINTER = 0.80


def build_model_from_checkpoint(path: str, device) -> SequenceVAE:
    payload = torch.load(path, map_location=device, weights_only=False)
    model_cfg = payload["model_cfg"]
    model = SequenceVAE(
        d_model=model_cfg["d_model"], nhead=model_cfg["nhead"],
        dim_feedforward=model_cfg["dim_feedforward"], dropout=model_cfg["dropout"],
        encoder_layers=model_cfg["encoder_layers"], decoder_layers=model_cfg["decoder_layers"],
        max_len=model_cfg["max_len"], d_z=model_cfg["d_z"],
        decoder_d_model=model_cfg.get("decoder_d_model"),
        decoder_nhead=model_cfg.get("decoder_nhead"),
        decoder_dim_feedforward=model_cfg.get("decoder_dim_feedforward"),
        length_head=model_cfg.get("length_head", "regression"),
        length_hidden=model_cfg.get("length_hidden", 256),
    )
    load_checkpoint(path, model, map_location=device)
    return model.to(device)


def diagnostic_pointers(report: dict) -> list[str]:
    pointers = []
    recon = report["reconstruction"]
    fa = report["faithfulness"]
    if not fa["latent_round_trip"]["pass"]:
        pointers.append(
            "Decoder does not render z faithfully (low round-trip R^2): the decoded "
            "sequence re-encodes to a different latent. Raise free_bits (the strongest "
            "lever found so far), or lower lipschitz_weight -- over-smoothing trades "
            "round-trip fidelity away directly."
        )
    if not fa["decode_non_collapse"]["pass"]:
        pointers.append(
            "Decodes are collapsing onto few or degenerate sequences, which also makes "
            "the round-trip score meaningless. Treat as posterior collapse: raise "
            "free_bits, lower beta_target, or extend KL annealing."
        )
    if not fa["z_utilization"]["pass"]:
        pointers.append(
            "Position accuracy from true z barely beats the length-matched mean latent: "
            "the decoder is behaving as a length-conditioned prior and largely ignoring "
            "per-sequence content in z. Raise free_bits and check the KL is not pinned "
            "at the per-dim floor across all dimensions."
        )
    if not fa["noise_survival"]["pass"]:
        pointers.append(
            "Decodes degenerate under latent noise at the scale a diffusion sampler will "
            "leave behind: the decoder is only valid on the exact encoder manifold. Raise "
            "lipschitz_weight (recalibrated 'normalized' mode) to flatten the decoder."
        )
    if recon.get("length_exact_rate", 1.0) < LENGTH_EXACT_RATE_POINTER:
        pointers.append(
            f"Length head gets the exact length right only {recon['length_exact_rate']:.0%} of the time "
            "(mean error {:.1f} residues); exact match cannot exceed that rate, so fix length first.".format(recon["length_mae"])
        )
    pc = report["posterior_collapse"]
    if not pc["prior_decode_diversity"]["pass"]:
        pointers.append(
            "High identical-decode rate from prior samples: likely posterior collapse. "
            "Raise free_bits, lower beta_target, or extend KL annealing duration."
        )
    if not pc["perturbation_sensitivity"]["pass"]:
        pointers.append(
            "Perturbation sensitivity flat or non-monotonic: decoder may be ignoring z "
            "locally, or lipschitz_weight may be too high (over-smoothing). Try adjusting "
            "lipschitz_weight or free_bits."
        )
    dh = report["density_holes"]
    if not dh["prior_sample_nondegeneracy"]["pass"]:
        pointers.append(
            "Degenerate decodes from broad prior sampling: aggregate posterior likely too "
            "narrow relative to the prior. Raise beta_target or extend annealing."
        )
    if not dh["aggregate_posterior_vs_prior"]["pass"]:
        pointers.append(
            "Aggregate posterior diverges from the prior (high MMD / per-dim deviation): "
            "holes in the latent space diffusion will undersample. Raise beta_target, "
            "consider a smaller d_z (forces denser packing), or extend pretraining."
        )
    sm = report["smoothness"]
    if not sm["interpolation_walks"]["pass"]:
        pointers.append(
            "Interpolation walks show large jumps or degenerate intermediate decodes: "
            "raise lipschitz_weight, or lower d_z to force denser latent coverage."
        )
    if not sm["nearest_neighbor_consistency"]["pass"]:
        pointers.append(
            "Latent-space nearness doesn't predict decoded-sequence nearness: consider "
            "raising lipschitz_weight or revisiting attention-pooling capacity."
        )
    return pointers


def run_freeze_report(model: SequenceVAE, split_sequences: list[str], train_sequences: list[str], device) -> dict:
    recon = reconstruction_fidelity(model, split_sequences, device)
    fa = faithfulness_check(model, split_sequences, train_sequences, device)
    pc = posterior_collapse_check(model, split_sequences, device)
    dh = density_holes_check(model, train_sequences, device)
    sm = smoothness_check(model, split_sequences, train_sequences, device)

    report = {
        # reported, not gated: exact match is unreachable for this decoder and
        # conflates encoder loss with decoder error -- `faithfulness` gates instead
        "reconstruction": recon,
        "faithfulness": fa,
        "posterior_collapse": pc,
        "density_holes": dh,
        "smoothness": sm,
        # reported, not gated: what a conditional stage-2 model depends on
        "properties": property_preservation(model, split_sequences, device),
        "prior_samples": prior_sample_properties(model, train_sequences, device),
        "gate_results": {
            "faithfulness": fa["pass"],
            "posterior_collapse": pc["pass"],
            "density_holes": dh["pass"],
            "smoothness": sm["pass"],
        },
    }
    report["overall_pass"] = all(report["gate_results"].values())
    report["diagnostic_pointers"] = diagnostic_pointers(report) if not report["overall_pass"] else []
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-config", default="config/data.yaml")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = get_device()
    model = build_model_from_checkpoint(args.checkpoint, device)

    data_cfg = load_yaml(args.data_config)
    df = pd.read_csv(data_cfg["finetune_clean"])
    split_sequences = df[df["split"] == args.split]["sequence"].tolist()
    train_sequences = df[df["split"] == "train"]["sequence"].tolist()

    print(f"running freeze report on split={args.split} (n={len(split_sequences)}), checkpoint={args.checkpoint}")
    report = run_freeze_report(model, split_sequences, train_sequences, device)

    print(f"overall_pass: {report['overall_pass']}")
    for gate, passed in report["gate_results"].items():
        print(f"  {gate}: {'PASS' if passed else 'FAIL'}")
    fa, recon_summary = report["faithfulness"], report["reconstruction"]
    rt, nc, zu, ns = fa["latent_round_trip"], fa["decode_non_collapse"], fa["z_utilization"], fa["noise_survival"]
    print("  faithfulness detail")
    print(f"    round trip: per-dim R^2 {rt['per_dim_r2']:.3f}, whitened |z-z_hat| {rt['mean_whitened_distance']:.2f} "
          f"vs {rt['random_real_pair_distance']:.2f} for a random real pair (ratio {rt['distance_ratio']:.3f})")
    print(f"    non-collapse: unique {nc['unique_frac']:.1%}, degenerate {nc['degenerate_frac']:.1%}")
    print(f"    z utilization: posAcc {zu.get('position_accuracy_from_z', float('nan')):.3f} from z vs "
          f"{zu.get('position_accuracy_from_length_mean', float('nan')):.3f} from the length-mean latent "
          f"(ratio {zu.get('ratio', float('nan')):.1f}x)")
    print(f"    noise survival: degenerate {ns['by_noise_level'][str(ns['gate_noise_frac'])]['degenerate_frac']:.1%} "
          f"at {ns['gate_noise_frac']} std noise")

    pr, ps = report["properties"], report["prior_samples"]
    print("  (reported, not gated)")
    print(f"  reconstruction: exact {recon_summary['exact_match_rate']:.1%}, edit {recon_summary['mean_edit_distance']:.3f}, "
          f"length exact {recon_summary['length_exact_rate']:.1%}")
    print(f"  property preservation (r): length {pr['length']['pearson_r']:.3f}, net charge {pr['net_charge']['pearson_r']:.3f}, "
          f"GRAVY {pr['gravy']['pearson_r']:.3f}; composition cosine {pr['composition_cosine']['matched']:.2f} "
          f"(shuffled baseline {pr['composition_cosine']['shuffled_baseline']:.2f})")
    print(f"  prior samples: charge {ps['samples']['net_charge'][0]:+.2f} vs real {ps['real']['net_charge'][0]:+.2f}, "
          f"GRAVY {ps['samples']['gravy'][0]:+.2f} vs {ps['real']['gravy'][0]:+.2f}, unique {ps['unique_frac']:.0%}, degenerate {ps['degenerate_frac']:.1%}")
    for pointer in report["diagnostic_pointers"]:
        print(f"  -> {pointer}")

    out_path = args.out or f"reports/freeze_report_{args.split}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    def strip_previews(d):
        # decoded_samples preview isn't JSON-critical and can be large; keep it small
        if isinstance(d, dict):
            return {k: strip_previews(v) for k, v in d.items()}
        if isinstance(d, list):
            return d[:20] if len(d) > 20 else d
        return d

    Path(out_path).write_text(json.dumps(strip_previews(report), indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
