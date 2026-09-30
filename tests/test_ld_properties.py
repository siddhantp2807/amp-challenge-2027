"""Regression test: the stage-2 conditioning properties must reproduce the labels.

data/ld-processed/{pretraining,finetuning}.csv were produced by
data-curation/scripts/feature_add/add_pc_features.py, which uses peptidy for
charge and modlamp for gravy/hydrophobic_moment. The diffusion model conditions
on those columns, so src/eval/properties.py has to agree with them *exactly* --
a constant offset between the training labels and the eval metric would show up
as model error that no amount of tuning removes.

Both upstream libraries use tables that differ from the published values
(peptidy's pKa set is not textbook Bjellqvist; modlamp's Eisenberg scale is
rounded to 2 s.f.), which is precisely why this is pinned rather than trusted.

Run: uv run python tests/test_ld_properties.py     (or under pytest)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.properties import charge_bjellqvist, gravy, hydrophobic_moment  # noqa: E402

LD = Path(__file__).resolve().parents[1] / "data" / "ld-processed"
# labels are float64 round-trips through CSV; atol here is well inside that noise
ATOL = 1e-9
COLUMNS = {
    "charge": charge_bjellqvist,
    "gravy": gravy,
    "hydrophobic_moment": hydrophobic_moment,
}


def check_file(path: Path) -> None:
    df = pd.read_csv(path)
    assert (df["length"] == df["sequence"].str.len()).all(), f"{path.name}: length column mismatch"

    for col, fn in COLUMNS.items():
        if col not in df.columns:
            continue
        got = np.array([fn(s) for s in df["sequence"]])
        want = df[col].to_numpy()
        err = np.abs(got - want)
        bad = int((err > ATOL).sum())
        assert bad == 0, (
            f"{path.name}:{col} -- {bad}/{len(df)} rows off by more than {ATOL:g} "
            f"(max {err.max():.3e}). The reimplementation has drifted from the "
            f"table the labels were generated with."
        )
        print(f"  {path.name:16s} {col:19s} n={len(df):6d}  max_err={err.max():.2e}  OK")


def test_ld_properties_match_labels():
    files = sorted(LD.glob("*.csv"))
    assert files, f"no label files under {LD}"
    for f in files:
        check_file(f)


def test_hydrophobic_moment_analytic():
    """Pin scale, angle and normalization without any external reference."""
    # a full helical turn of one residue type cancels: 100 deg * 18 = 5 turns
    assert hydrophobic_moment("A" * 18) < 0.02
    # an idealized amphipathic helix concentrates the moment
    assert hydrophobic_moment("LKKLLKLLKKLLKLLKKL") > 0.4
    # the 3.6-residue helical period is what 100 deg selects; a 2-periodic
    # (beta-strand) pattern scores far higher at the 160 deg sheet angle
    strand = "LKLKLKLKLKLKLKLKLK"
    assert hydrophobic_moment(strand, angle=160.0) > hydrophobic_moment(strand, angle=100.0)
    assert hydrophobic_moment("") == 0.0


def test_charge_definitions_differ():
    """Guard against silently swapping in the rule-based net_charge.

    The two correlate at r=1.0000 and differ by a near-constant ~0.26 (mostly the
    termini terms net_charge omits), so no single example proves anything and
    rankings are unaffected. What it would do is add a systematic ~0.086 std bias
    to reported charge MAE -- about 44% of the 0.195 std of headroom that exists
    between the conditional-Gaussian baseline (0.744) and the decoder ceiling
    (0.549). That is why stage 2 pins the definition instead of treating them as
    interchangeable.
    """
    from src.eval.properties import net_charge
    df = pd.read_csv(LD / "finetuning.csv")
    bj = np.array([charge_bjellqvist(s) for s in df.sequence])
    rule = np.array([net_charge(s) for s in df.sequence])
    bias = np.abs(bj - rule).mean() / df["charge"].std()
    assert 0.05 < bias < 0.15, f"charge-definition bias {bias:.3f} std moved unexpectedly"


if __name__ == "__main__":
    test_ld_properties_match_labels()
    test_hydrophobic_moment_analytic()
    test_charge_definitions_differ()
    print("all property regression checks passed")
