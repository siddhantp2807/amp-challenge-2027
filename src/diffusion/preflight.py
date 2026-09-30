"""Feasibility gates that run BEFORE any diffusion training.

The point is to fail fast and cheaply. Conditioning cannot beat what the latent
actually encodes, so if a property is poorly recoverable from z, no amount of
diffusion tuning will fix it -- that is stage-1 work. This is exactly why
hydrophobic moment was dropped in favour of GRAVY (MLP R^2 0.639 vs 0.930).

Gates:
  G1 decodability   MLP R^2 per conditioned property from z
  G2 linearity      ridge R^2 -- informational; low values justify Fourier features
  G3 ceiling        property MAE from decoding the TRUE val latents
  G4 baseline       conditional-Gaussian MAE; diffusion must beat this
  G5 support        fraction of val condition triples inside the training joint
  G6 posterior      is z ~ q(z|x) real augmentation, or cosmetic?

Writes reports/diffusion_preflight.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.diffusion.baseline import ConditionalGaussian, ConditionalGMM, UnconditionalGaussian
from src.diffusion.conditioning import PROPERTIES
from src.diffusion.data import build_split_frames, encode_frame
from src.diffusion.latents import load_vae, posterior_width_report
from src.diffusion.metrics import score_latents
from src.train.utils import get_device, load_yaml, set_seed

R2_PROCEED = 0.60
R2_CONFIDENT = 0.75
SUPPORT_MIN = 0.95


def _r2(pred, true):
    return float(1 - ((true - pred) ** 2).sum() / ((true - true.mean()) ** 2).sum())


def probe_decodability(z_tr, c_tr, z_va, c_va, seed: int = 0) -> dict:
    """How much of each property is recoverable from z, linearly and nonlinearly."""
    from sklearn.linear_model import RidgeCV
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    out = {}
    xs = StandardScaler().fit(z_tr)
    ztr, zva = xs.transform(z_tr), xs.transform(z_va)
    for i, name in enumerate(PROPERTIES):
        ys = StandardScaler().fit(c_tr[:, [i]])
        ytr = ys.transform(c_tr[:, [i]]).ravel()
        yva = c_va[:, i]
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(ztr, ytr)
        mlp = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=600,
                           random_state=seed, early_stopping=True).fit(ztr, ytr)
        r_ridge = _r2(ys.inverse_transform(ridge.predict(zva).reshape(-1, 1)).ravel(), yva)
        r_mlp = _r2(ys.inverse_transform(mlp.predict(zva).reshape(-1, 1)).ravel(), yva)
        out[name] = {
            "ridge_r2": r_ridge,
            "mlp_r2": r_mlp,
            "pass": bool(r_mlp >= R2_PROCEED),
            "confident": bool(r_mlp >= R2_CONFIDENT),
        }
    out["pass"] = all(v["pass"] for k, v in out.items() if k in PROPERTIES)
    return out


def support_gate(c_tr: np.ndarray, c_va: np.ndarray) -> dict:
    """Are the requested condition triples inside the training joint?

    charge and GRAVY are anti-correlated (r about -0.45 on the target set), so the
    condition space is not a box: 'high charge AND high GRAVY' is off-manifold and
    will generate poorly. This makes the gate load-bearing rather than a formality.
    """
    mean, cov = c_tr.mean(0), np.cov(c_tr, rowvar=False)
    inv = np.linalg.inv(cov)
    d = c_va - mean
    maha = np.sqrt(np.einsum("ij,jk,ik->i", d, inv, d))
    frac = float((maha < 3).mean())
    return {
        "mahalanobis_median": float(np.median(maha)),
        "frac_within_3": frac,
        "condition_correlations": np.corrcoef(c_tr, rowvar=False).round(4).tolist(),
        "pass": bool(frac >= SUPPORT_MIN),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-checkpoint", default="checkpoints/release/finetune_fb3p0_lip0p1_v1_best.pt")
    ap.add_argument("--data-config", default="config/data.yaml")
    ap.add_argument("--out", default="reports/diffusion_preflight.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline-seeds", type=int, default=5)
    ap.add_argument("--gmm-components", type=int, nargs="+", default=[4, 8, 16])
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    vae = load_vae(args.vae_checkpoint, device)
    frames = build_split_frames(load_yaml(args.data_config))
    tr = encode_frame(vae, frames["train"], device)
    va = encode_frame(vae, frames["val"], device)

    z_tr, z_va = tr.mu.numpy(), va.mu.numpy()
    c_tr, c_va = tr.conditions, va.conditions
    train_std = c_tr.std(0)
    rng = np.random.default_rng(args.seed)

    report = {
        "vae_checkpoint": args.vae_checkpoint,
        "properties": list(PROPERTIES),
        "n_train": len(tr), "n_val": len(va),
        "train_std": dict(zip(PROPERTIES, train_std.round(4).tolist())),
    }

    print("G1/G2 decodability probe ...")
    report["G1_G2_decodability"] = probe_decodability(z_tr, c_tr, z_va, c_va, args.seed)
    for p in PROPERTIES:
        d = report["G1_G2_decodability"][p]
        print(f"  {p:8s} ridge R2 {d['ridge_r2']:.3f}  MLP R2 {d['mlp_r2']:.3f}"
              f"  {'OK' if d['pass'] else 'FAIL'}")

    print("G3 decoder ceiling ...")
    report["G3_decoder_ceiling"] = score_latents(
        vae, va.mu, c_va, train_std, tr.sequences, device)
    print("  ", {k: round(v, 3) for k, v in
                 report["G3_decoder_ceiling"]["property_mae"].items() if isinstance(v, float)})

    print("G4 baselines ...")
    uncond = UnconditionalGaussian.fit(z_tr)
    cg = ConditionalGaussian.fit(z_tr, c_tr)
    baselines = {
        "unconditional_gaussian": uncond.sample(len(c_va), rng),
        "conditional_gaussian": cg.sample(c_va, rng),
        "conditional_gaussian_mean_only": cg.conditional_mean(c_va),
    }
    # Sweep K rather than trusting one setting: a K-component mixture over the
    # 67-d joint has ~K*2.3k free parameters against 6,690 training points, so the
    # bar must not be an artifact of an over-parameterized fit. Scored on val, so
    # overfitting shows up as a *worse* number rather than a flattering one.
    for k in args.gmm_components:
        try:
            gmm = ConditionalGMM.fit(z_tr, c_tr, k, args.seed)
            baselines[f"conditional_gmm_k{k}"] = gmm.sample(c_va, rng)
        except Exception as e:                                # sklearn/scipy optional
            print(f"  (skipping GMM K={k}: {e})")

    report["G4_baselines"] = {}
    for name, z in baselines.items():
        report["G4_baselines"][name] = score_latents(
            vae, z, c_va, train_std, tr.sequences, device)
        m = report["G4_baselines"][name]["property_mae"]
        print(f"  {name:32s}", {p: round(m[p], 3) for p in PROPERTIES},
              f"uniq={report['G4_baselines'][name]['unique_frac']:.3f}",
              f"copy={report['G4_baselines'][name].get('copy_frac', float('nan')):.3f}")

    # Error bars. A single draw of 372 samples is not enough to rank these: an
    # early single-seed run had GMM K=8 at charge 0.708 and a re-run put it at
    # 0.839, i.e. the spread between methods on charge is comparable to the spread
    # between draws of the same method. Repeat each sampler and report mean +/- sd,
    # so "diffusion beat the bar" cannot be a lucky seed.
    print(f"G4b baseline stability over {args.baseline_seeds} seeds ...")
    stability = {}
    samplers = {"conditional_gaussian": lambda r: cg.sample(c_va, r)}
    for k in args.gmm_components:
        try:
            g = ConditionalGMM.fit(z_tr, c_tr, k, args.seed)
            samplers[f"conditional_gmm_k{k}"] = (lambda gg: lambda r: gg.sample(c_va, r))(g)
        except Exception:
            pass
    for name, fn in samplers.items():
        runs = []
        for s in range(args.baseline_seeds):
            zz = fn(np.random.default_rng(1000 + s))
            mm = score_latents(vae, zz, c_va, train_std, tr.sequences, device)["property_mae"]
            runs.append([mm[p] for p in PROPERTIES])
        arr = np.array(runs)
        stability[name] = {
            "mean": dict(zip(PROPERTIES, arr.mean(0).round(4).tolist())),
            "sd": dict(zip(PROPERTIES, arr.std(0).round(4).tolist())),
            "n_seeds": args.baseline_seeds,
        }
        print(f"  {name:24s} " + "  ".join(
            f"{p}={arr.mean(0)[i]:.3f}+/-{arr.std(0)[i]:.3f}" for i, p in enumerate(PROPERTIES)))
    report["G4b_baseline_stability"] = stability

    # The primary bar. p(z|c) turned out to be multimodal -- the mixture beats the
    # linear-Gaussian well outside noise -- so "beat the conditional Gaussian" is
    # too weak a target. Pick the best GMM by mean MAE across properties.
    gmm_rows = {k: v for k, v in stability.items() if k.startswith("conditional_gmm")}
    if gmm_rows:
        best = min(gmm_rows, key=lambda k: np.mean(
            [gmm_rows[k]["mean"][p] for p in PROPERTIES]))
        report["primary_bar"] = {
            "name": best,
            "property_mae": gmm_rows[best]["mean"],
            "property_mae_sd": gmm_rows[best]["sd"],
            "rationale": (
                "Conditional GMM, not the linear-Gaussian: the mixture beats it on every "
                "property, so p(z|c) is multimodal and a single ellipsoid per condition "
                "under-fits. Diffusion must beat THIS to be worth its parameters."
            ),
        }
        print(f"  primary bar = {best}: "
              f"{ {p: round(report['primary_bar']['property_mae'][p], 3) for p in PROPERTIES} }")

    print("G5 support ...")
    report["G5_support"] = support_gate(c_tr, c_va)
    print(f"  frac within Mahalanobis 3: {report['G5_support']['frac_within_3']:.3f}")

    print("G6 posterior width ...")
    report["G6_posterior_width"] = posterior_width_report(tr.logvar, tr.mu.std(0))
    g6 = report["G6_posterior_width"]
    print(f"  ratio {g6['ratio']:.3f} -> "
          f"{'real' if g6['augmentation_is_meaningful'] else 'COSMETIC'} augmentation")

    report["overall_pass"] = bool(
        report["G1_G2_decodability"]["pass"] and report["G5_support"]["pass"]
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\noverall_pass={report['overall_pass']} -> {args.out}")


if __name__ == "__main__":
    main()
