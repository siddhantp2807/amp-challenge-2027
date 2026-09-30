"""Regression test: MIC censoring must arrive in the label file flipped into pMIC.

DBAASP records censoring in MIC space. ">128 µM" is a RIGHT-censored MIC and means
"no activity up to 128" -- an inactive peptide. pMIC = -log10(MIC in M) is
DECREASING in MIC, so the same observation is a LEFT-censored pMIC.

If that flip is inverted, the Tobit likelihood is told the 1,749 inactive cells are
the most potent ones. Nothing downstream complains: per-head Spearman is computed on
the exact cells, which are still fit correctly, so the number stays plausible while
the ranking the submission ships is driven by an inverted weak tail.

Hence two independent checks, on the shipped label file:

  1. a round trip on the raw frame -- the specific rows DBAASP marks right-censored
     must come out as censor == -1;
  2. a distributional check -- left-censored cells are the inactive tail, so their
     median pMIC must sit below the exact cells'.

Run: uv run python tests/test_censoring_sign.py     (or under pytest)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIC = ROOT / "data" / "processed-data" / "dbaasp-std" / "filtered-dbaasp.csv"
LABELS = ROOT / "data" / "processed-data" / "mic-labels" / "mic_labels.csv"


def test_right_censored_mic_becomes_left_censored_pmic():
    """A cell whose only observations are ">X" must land as censor == -1."""
    raw = pd.read_csv(MIC)
    raw["mod_concentration"] = pd.to_numeric(raw["mod_concentration"], errors="coerce")
    raw = raw[raw["mod_concentration"] > 0]

    # Cells with no exact measurement at all -- those are the ones whose censoring
    # survives collapse(). A cell with any exact row is stored as that value.
    per_cell = raw.groupby(["sequence", "species_clean"])["censor_direction"]
    only_right = per_cell.apply(lambda s: s.notna().all() and (s == "right").all())
    only_right = set(only_right[only_right].index)
    assert only_right, "no purely right-censored cells found; fixture is wrong"

    labels = pd.read_csv(LABELS)
    got = labels.set_index(["sequence", "species"])["censor"].to_dict()

    checked = 0
    for key in only_right:
        if key not in got:          # species outside the 6-head panel
            continue
        assert got[key] == -1, (
            f"{key}: right-censored MIC stored as censor={got[key]}, expected -1. "
            "The MIC->pMIC censoring flip is inverted."
        )
        checked += 1
    assert checked > 100, f"only {checked} cells checked; expected hundreds"
    print(f"  right-censored MIC -> censor=-1 on {checked} cells   OK")


def test_left_censored_cells_are_the_weaker_tail():
    """Inactive peptides must be less potent than measured ones, in pMIC space."""
    labels = pd.read_csv(LABELS)
    med_exact = labels.loc[labels["censor"] == 0, "y_pmic"].median()
    med_left = labels.loc[labels["censor"] == -1, "y_pmic"].median()
    assert med_left < med_exact, (
        f"left-censored median pMIC {med_left:.3f} is not below exact "
        f"{med_exact:.3f}; the censoring sign looks inverted."
    )
    print(f"  median pMIC  exact {med_exact:.3f} > left-censored {med_left:.3f}   OK")


def test_pmic_is_decreasing_in_mic():
    """Guards the y_pmic formula itself, independent of censoring."""
    raw = pd.read_csv(MIC)
    raw["mod_concentration"] = pd.to_numeric(raw["mod_concentration"], errors="coerce")
    exact = raw[(raw["mod_concentration"] > 0) & raw["censor_direction"].isna()]
    exact = exact[exact["species_clean"] == "Escherichia coli"]
    # The median is taken in pMIC space, not MIC space: for an even number of
    # replicates the two differ (log of a mean is not the mean of logs), and the
    # modelling space is the one that should be summarised.
    exact = exact.assign(pmic=6.0 - np.log10(exact["mod_concentration"]))
    per_seq = exact.groupby("sequence")["pmic"].median()
    per_seq_mic = exact.groupby("sequence")["mod_concentration"].median()

    labels = pd.read_csv(LABELS)
    ec = labels[(labels["species"] == "Escherichia coli") & (labels["censor"] == 0)]
    ec = ec.set_index("sequence")["y_pmic"]

    common = per_seq.index.intersection(ec.index)
    assert len(common) > 1000
    assert np.allclose(per_seq.loc[common].values, ec.loc[common].values, atol=1e-9), (
        "y_pmic does not equal the median of 6 - log10(MIC in µM)"
    )
    rho = np.corrcoef(per_seq_mic.loc[common], ec.loc[common])[0, 1]
    assert rho < 0, f"pMIC should decrease with MIC, got correlation {rho:+.3f}"
    print(f"  y_pmic == median(6 - log10 MIC) on {len(common)} cells, "
          f"corr vs MIC {rho:+.3f}   OK")


if __name__ == "__main__":
    test_right_censored_mic_becomes_left_censored_pmic()
    test_left_censored_cells_are_the_weaker_tail()
    test_pmic_is_decreasing_in_mic()
    print("\nall censoring-sign checks passed")
