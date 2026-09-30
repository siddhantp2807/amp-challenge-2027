#!/usr/bin/env python3
"""
05-standardize-mic.py
=====================
STEP 5 of the DBAASP pipeline.

Turns the activity table from 04 into a standardised sequence x species x MIC
frame: monomer MIC rows only, concentrations parsed out of free text, everything
converted to µM, target strains collapsed onto a fixed species panel, and the
competition's own sequence gates applied.

Ported from ../data-curation/notebooks/22-dbaasp-std.ipynb (section 2 onward;
section 1 there rebuilds the activity table from the detail JSONs, which is what
our 04 script already does). With the default --medium MHB this reproduces that
notebook's filtered-dbaasp-2344.csv exactly: 10,663 rows, 2,344 unique sequences.

TWO DELIBERATE DEPARTURES
-------------------------
1. `is_censored` / `censor_direction` columns are added. The notebook strips ">"
   and keeps the bound as a plain number, which silently turns "no activity up to
   128 µM" into "MIC = 128 µM" -- 25% of rows. The numeric values here are left
   byte-identical to the notebook so the output still reproduces, but the fact of
   censoring is preserved so a downstream Tobit likelihood can use it.
2. The notebook's manual fixes were positional (`.drop(index=[177572])`,
   `.at[180026, ...]`), which break the moment an upstream row count changes.
   They are rewritten here as value-based predicates doing the same thing.

MEDIUM
------
The notebook keeps `medium == 'MHB'` only. That is defensible for assay
comparability but costs coverage: MHB alone covers 27.6% of this project's
training peptides, MHB+CAMHB 31.3%, any medium 50.1%. CAMHB is cation-adjusted
MHB, the CLSI reference broth. --medium is therefore a list, defaulting to MHB
for reproduction; widen it for the library weighting step.

USAGE
-----
    python 05-standardize-mic.py                                  # reproduces the notebook
    python 05-standardize-mic.py --medium MHB CAMHB               # wider coverage
    python 05-standardize-mic.py --medium ANY --out wide.csv      # no medium filter
    python 05-standardize-mic.py --verify ../data-curation/.../filtered-dbaasp-2344.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from peptidy.descriptors import molecular_weight

DEFAULT_ACTIVITY_CSV = Path("data/raw-data/dbaasp/dbaasp-activity.csv")
DEFAULT_OUT = Path("data/processed-data/dbaasp-std/filtered-dbaasp.csv")

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Substring-matched against target_species, which carries strain designations
# ("Escherichia coli ATCC 25922"). 8,239 distinct raw strings collapse onto these.
PANEL_SPECIES = [
    "Acinetobacter baumannii",
    "Enterobacter cloacae",
    "Escherichia coli",
    "Klebsiella pneumoniae",
    "Pseudomonas aeruginosa",
    "Salmonella enterica",
    "Staphylococcus aureus",
    "Enterococcus faecalis",
    "Enterococcus faecium",
]

KEEP_COLUMNS = [
    "dbaaspId", "sequence", "target_species", "activity_measure_value",
    "concentration", "unit", "medium", "cTerminus", "nTerminus",
    "intrachainBondCount",
]

# Right/left censoring markers, longest first so ">=" is not read as ">".
RIGHT_MARKERS = (">=", ">>", "=>", "≥", ">")
LEFT_MARKERS = ("<=", "≤", "<")


def calculate_molecular_weight(sequence):
    """Molecular weight in Da, or None for sequences peptidy cannot parse."""
    if not isinstance(sequence, str):
        return None
    sequence = sequence.strip().upper().replace(" ", "").replace("\n", "")
    if not sequence:
        return None
    try:
        return molecular_weight(sequence)
    except (ValueError, KeyError):
        return None


# Each row is grouped by the exact list of non-numeric character runs in its
# concentration string, and each group gets its own cleaner: keep the lower bound
# of ranges, drop ± error terms, drop </> qualifiers. Ported verbatim from the
# notebook (which has no key 27). Patterns not listed stay None and drop at the
# float cast below.
PATTERN_DICT = {
    1:  {"pattern": "['>']", "function": lambda x: x.replace(">", "").strip()},
    2:  {"pattern": "['-']", "function": lambda x: x.split("-")[0]},
    3:  {"pattern": "['±']", "function": lambda x: x.split("±")[0].strip()},
    4:  {"pattern": "['<']", "function": lambda x: x.replace("<", "").strip()},
    5:  {"pattern": "['>=']", "function": lambda x: x.replace(">=", "").strip()},
    6:  {"pattern": "['<=']", "function": lambda x: x.replace("<=", "").strip()},
    7:  {"pattern": "[' ']", "function": lambda x: x.strip()},
    8:  {"pattern": "['->']", "function": lambda x: x.split("->")[0]},
    9:  {"pattern": "['>', ' ']", "function": lambda x: x.strip().replace(">", "").strip()},
    10: {"pattern": "[' ± ']", "function": lambda x: x.split("±")[0].strip()},
    11: {"pattern": "['>>']", "function": lambda x: x.replace(">>", "").strip()},
    12: {"pattern": "['<=', '-']", "function": lambda x: x.replace("<=", "").split("-")[0].strip()},
    13: {"pattern": "['<=', '±']", "function": lambda x: x.replace("<=", "").split("±")[0].strip()},
    14: {"pattern": "['±', ' ']", "function": lambda x: x.split("±")[0].strip()},
    15: {"pattern": "['≥']", "function": lambda x: x.replace("≥", "").strip()},
    16: {"pattern": "['≥ ']", "function": lambda x: x.replace("≥", "").strip()},
    17: {"pattern": "['–']", "function": lambda x: x.split("–")[0].strip()},
    18: {"pattern": "[',']", "function": lambda x: x.replace(",", ".").strip()},
    19: {"pattern": "[' ±']", "function": lambda x: x.split("±")[0].strip()},
    20: {"pattern": "['>', '  ']", "function": lambda x: x.replace(">", "").strip()},
    21: {"pattern": "['\\t']", "function": lambda x: x.strip()},
    22: {"pattern": "['>', '-']", "function": lambda x: x.replace(">", "").split("-")[0]},
    23: {"pattern": "['>', '\\t']", "function": lambda x: x.replace(">", "").strip()},
    24: {"pattern": "[' ', '±']", "function": lambda x: x.split("±")[0].strip()},
    25: {"pattern": "['->=']", "function": lambda x: x.split("->=")[0].strip()},
    26: {"pattern": "['-', ' ']", "function": lambda x: x.split("-")[0].strip()},
    28: {"pattern": "['± ']", "function": lambda x: x.split("±")[0].strip()},
    29: {"pattern": "['<', '-']", "function": lambda x: x.replace("<", "").split("-")[0]},
    30: {"pattern": "['>>', '±']", "function": lambda x: x.replace(">>", "").split("±")[0]},
    31: {"pattern": "[' ', ' ']", "function": lambda x: x.strip()},
    32: {"pattern": "[' ', '-']", "function": lambda x: x.split("-")[0].strip()},
    33: {"pattern": "[' ', ' ± ']", "function": lambda x: x.split("±")[0].strip()},
    34: {"pattern": "['-<']", "function": lambda x: x.split("-")[0].strip()},
    35: {"pattern": "['<', '±']", "function": lambda x: x.replace("<", "").split("±")[0]},
    36: {"pattern": "['-', '\\t']", "function": lambda x: x.split("-")[0]},
    37: {"pattern": "[' - >=']", "function": lambda x: x.split("-")[0].strip()},
    38: {"pattern": "['–>']", "function": lambda x: x.split("–")[0]},
    39: {"pattern": "['      ']", "function": lambda x: x.strip()},
    40: {"pattern": "[' - >']", "function": lambda x: x.split("-")[0].strip()},
    41: {"pattern": "[' - =>', '\\t']", "function": lambda x: x.split("-")[0].strip()},
    42: {"pattern": "[' - =>']", "function": lambda x: x.split("-")[0].strip()},
    43: {"pattern": "['–', '\\t']", "function": lambda x: x.split("–")[0]},
    44: {"pattern": "[' ± ', ' ']", "function": lambda x: x.split("±")[0].strip()},
    45: {"pattern": "[' ±', ' ']", "function": lambda x: x.split("±")[0].strip()},
    46: {"pattern": "[' ', ' ± ', ' ']", "function": lambda x: x.split("±")[0].strip()},
    47: {"pattern": "['  ']", "function": lambda x: x.strip()},
    48: {"pattern": "['+']", "function": lambda x: x.split("+")[0]},
}


def censor_direction(raw):
    """'right' | 'left' | None, read off the raw concentration string."""
    if not isinstance(raw, str):
        return None
    if any(m in raw for m in RIGHT_MARKERS):
        return "right"
    if any(m in raw for m in LEFT_MARKERS):
        return "left"
    return None


def clean_concentrations(df):
    """Parse free-text concentration into a float, grouping by punctuation shape."""
    df = df.copy()
    df["concentration_spl_chars"] = df["concentration"].str.findall(r"[^\d.]+").astype(str)
    df["cleaned_concentration"] = None

    plain = df.index[df["concentration_spl_chars"] == "[]"]
    df.loc[plain, "cleaned_concentration"] = df.loc[plain, "concentration"]

    for spec in PATTERN_DICT.values():
        idx = df.index[df["concentration_spl_chars"] == spec["pattern"]]
        if len(idx):
            df.loc[idx, "cleaned_concentration"] = df.loc[idx, "concentration"].apply(spec["function"])

    unparsed = df["cleaned_concentration"].isna().sum()
    if unparsed:
        print(f"  {unparsed} rows matched no cleaning pattern and will be dropped")
    return df


def apply_manual_fixes(df):
    """The notebook's positional fixes, restated as value-based predicates."""
    # DBAASPS_7342: sequence 'XXX', concentration '4.5.5' -- unparseable
    df = df[~((df["dbaaspId"] == "DBAASPS_7342") & (df["concentration"] == "4.5.5"))]
    df.loc[df["cleaned_concentration"] == "13. 8", "cleaned_concentration"] = "13.8"
    return df


def convert_to_micromolar(conc, unit, mol_wt):
    """MIC(µM) = MIC(µg/ml) * 1000 / MW(g/mol); other units pass through."""
    if unit == "µg/ml":
        if mol_wt is None or pd.isna(mol_wt) or mol_wt == 0:
            return None
        return conc * 1000 / mol_wt
    return conc


def assign_panel_species(df):
    """Collapse strain-level target_species onto the fixed panel."""
    df = df.copy()
    df["species_clean"] = None
    for species in PANEL_SPECIES:
        idx = df.index[df["target_species"].str.contains(species, regex=False, na=False)]
        df.loc[idx, "species_clean"] = species
    df = df.dropna(subset=["species_clean"])

    # Typhimurium is the most-tested S. enterica serovar; keep it as its own level
    typh = df.index[df["target_species"].str.contains("Typhimurium", regex=False, na=False)]
    df.loc[typh, "species_clean"] = "Salmonella enterica Typhimurium"
    return df


def apply_sequence_gates(df):
    """The competition's own sequence constraints (README 'Sequence Requirements')."""
    df = df[df["cTerminus"].isna() & df["nTerminus"].isna()]
    df = df[df["sequence"].notna()]
    df = df[df["sequence"].apply(lambda s: set(s).issubset(CANONICAL_AA))]
    lengths = df["sequence"].apply(len)
    df = df[(lengths >= 8) & (lengths <= 50)]
    df = df[df["intrachainBondCount"] == 0]
    return df


def standardize(activity_csv, media):
    df = pd.read_csv(activity_csv, low_memory=False)
    print(f"{len(df)} activity rows, {df['dbaaspId'].nunique()} unique dbaaspId")

    df = df[df["complexity"] == "Monomer"][KEEP_COLUMNS]
    df = df[df["activity_measure_value"] == "MIC"].copy()
    df = df.dropna(subset=["unit"]).dropna(subset=["concentration"])
    print(f"  monomer MIC rows with unit+concentration: {len(df)}")

    # record censoring from the RAW string, before cleaning strips the markers
    df["censor_direction"] = df["concentration"].apply(censor_direction)
    df["is_censored"] = df["censor_direction"].notna()

    df["mol_wt"] = df["sequence"].apply(calculate_molecular_weight)
    df = clean_concentrations(df)

    if media is not None:
        df = df[df["medium"].isin(media)]
        print(f"  after medium filter {media}: {len(df)}")

    df = apply_manual_fixes(df)
    df["cleaned_concentration"] = df["cleaned_concentration"].astype(float)

    df["mod_concentration"] = df.apply(
        lambda r: convert_to_micromolar(float(r["cleaned_concentration"]), r["unit"], r["mol_wt"]),
        axis=1,
    )
    df["mod_unit"] = "µM"

    # '0-X' ranges were cleaned to 0 by the lower-bound rule; use the midpoint
    df.loc[(df["dbaaspId"] == "DBAASPS_5009") & (df["concentration"] == "0-6.06"), "mod_concentration"] = 3.03
    df.loc[(df["dbaaspId"] == "DBAASPS_5010") & (df["concentration"] == "0-32.5"), "mod_concentration"] = 16.25
    zeros = (df["mod_concentration"] == 0).sum()
    if zeros:
        print(f"  WARNING: {zeros} rows still have MIC == 0")

    df = df.drop(columns=["concentration_spl_chars"]).reset_index(drop=True)
    df = df.dropna(subset=["mol_wt"])
    df = assign_panel_species(df)
    df = apply_sequence_gates(df)
    return df.reset_index(drop=True)


REFERENCE_COLUMNS = [
    "dbaaspId", "sequence", "target_species", "activity_measure_value", "concentration",
    "unit", "medium", "cTerminus", "nTerminus", "intrachainBondCount", "mol_wt",
    "cleaned_concentration", "mod_concentration", "mod_unit", "species_clean",
]


def verify(written_csv, reference_csv):
    """Check the port against the notebook's own output on their shared columns.

    Both sides are read back from disk rather than compared in memory: the
    notebook's frame also round-trips through CSV, and an in-memory object column
    of Nones does not stringify the same way as a float NaN read from a file.
    """
    df = pd.read_csv(written_csv)
    ref = pd.read_csv(reference_csv)
    ok = True
    if len(df) != len(ref):
        print(f"  ROW COUNT differs: {len(df)} vs reference {len(ref)}"); ok = False
    if df["sequence"].nunique() != ref["sequence"].nunique():
        print(f"  UNIQUE SEQ differs: {df['sequence'].nunique()} vs {ref['sequence'].nunique()}"); ok = False
    if ok:
        a = df[REFERENCE_COLUMNS].reset_index(drop=True)
        b = ref[REFERENCE_COLUMNS].reset_index(drop=True)
        for col in REFERENCE_COLUMNS:
            if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
                same = ((a[col] - b[col]).abs() < 1e-9) | (a[col].isna() & b[col].isna())
            else:
                same = (a[col].astype(str) == b[col].astype(str))
            if not same.all():
                print(f"  COLUMN '{col}' differs in {(~same).sum()} rows"); ok = False
    print("  VERIFY:", "PASS -- matches the reference notebook output" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--activity-csv", type=Path, default=DEFAULT_ACTIVITY_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--medium", nargs="+", default=["MHB"],
                    help="media to keep; 'ANY' disables the filter (default: MHB)")
    ap.add_argument("--verify", type=Path, default=None,
                    help="reference CSV to check the port against")
    args = ap.parse_args()

    media = None if [m.upper() for m in args.medium] == ["ANY"] else args.medium
    df = standardize(args.activity_csv, media)

    print(f"\n{len(df)} rows, {df['sequence'].nunique()} unique sequences")
    print(f"censored: {df['is_censored'].sum()} rows "
          f"({df['censor_direction'].value_counts().to_dict()})")
    print("\nspecies_clean:")
    print(df["species_clean"].value_counts().to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    if args.verify:
        print(f"\nverifying against {args.verify}")
        verify(args.out, args.verify)


if __name__ == "__main__":
    main()
