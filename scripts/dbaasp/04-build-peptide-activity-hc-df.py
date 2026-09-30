# Step 1: imports and CLI argument parsing.
import argparse
import json
from pathlib import Path

import pandas as pd

# the 13 physicoChemicalProperties names are fixed/confirmed identical across every
# peptide that has the property list at all -- used to keep pivoted columns consistent
# even for peptides missing the list entirely.
PHYSCHEM_PROPERTIES = [
    "Amphiphilicity Index",
    "Angle Subtended by the Hydrophobic Residues",
    "Disordered Conformation Propensity",
    "ID",
    "Isoelectric Point",
    "Linear Moment",
    "Net Charge",
    "Normalized Hydrophobic Moment",
    "Normalized Hydrophobicity",
    "Penetration Depth",
    "Propensity to PPII coil",
    "Propensity to in vitro Aggregation",
    "Tilt Angle",
]

# Step 2: shared dict-to-name flattening helper -- every {name, description}-shaped
# lookup dict in this schema (complexity, cTerminus, nTerminus, targetSpecies/targetCell,
# activityMeasureGroup, unit, medium, cfuGroup, ...) reduces to just its "name".
def flatten_name(d):
    return d["name"] if isinstance(d, dict) else None

# intrachainBonds lives at the top level for Monomer peptides, but only inside each
# monomers[] entry for Multimer/Multi-Peptide ones (never both). Collect from both,
# tagging monomer-level bonds with a 1-based "monomer" index (matching the order the
# monomer sequences get ","-joined in later), and serialise to one JSON string column
# so the bond list doesn't multiply activity rows.
def flatten_intrachain_bonds(record):
    def flatten_bond(b, monomer=None):
        bond = {
            "position1": b.get("position1"),
            "position2": b.get("position2"),
            "type": flatten_name(b.get("type")),
            "cycleType": flatten_name(b.get("cycleType")),
            "chainParticipating": flatten_name(b.get("chainParticipating")),
            "note": b.get("note") or None,
        }
        if monomer is not None:
            bond["monomer"] = monomer
        return bond

    bonds = [flatten_bond(b) for b in (record.get("intrachainBonds") or [])]
    for i, m in enumerate(record.get("monomers") or [], start=1):
        bonds.extend(flatten_bond(b, monomer=i) for b in (m.get("intrachainBonds") or []))

    return {
        "intrachainBonds": json.dumps(bonds) if bonds else None,
        "intrachainBondCount": len(bonds),
    }

# Step 3: build the peptide-level fields shared by every exploded row of one dbaaspId
# (id/name/sequence/url/complexity/termini/targetGroups/smiles/intrachainBonds + 13 physchem_ columns).
def build_peptide_shared_fields(record):
    fields = {
        "dbaaspId": record.get("dbaaspId"),
        "name": record.get("name"),
        "sequence": record.get("sequence"),
        "sequenceLength": record.get("sequenceLength"),
        "url": record.get("url"),
        "complexity": flatten_name(record.get("complexity")),
        "cTerminus": flatten_name(record.get("cTerminus")),
        "nTerminus": flatten_name(record.get("nTerminus")),
        "targetGroups": "; ".join(
            g["name"] for g in (record.get("targetGroups") or []) if g.get("name")
        ) or None,
        "smiles": "; ".join(
            s["smiles"] for s in (record.get("smiles") or []) if s.get("smiles")
        ) or None,
        **flatten_intrachain_bonds(record),
    }

    pcp_by_name = {p["name"]: p["value"] for p in (record.get("physicoChemicalProperties") or [])}
    for prop_name in PHYSCHEM_PROPERTIES:
        fields[f"physchem_{prop_name}"] = pcp_by_name.get(prop_name)

    return fields

# Step 4: explode one peptide's targetActivities into row-dicts, merged with its shared
# fields. A peptide with zero targetActivities still gets one row with blank activity columns.
ACTIVITY_BLANK = {
    "target_species": None, "activity_measure_group": None, "activity_measure_value": None,
    "concentration": None, "unit": None, "medium": None, "cfu": None, "cfu_group": None,
    "ph": None, "ionic_strength": None, "salt_type": None, "note": None, "reference": None,
    "activity": None,
}


def build_activity_rows(record, shared):
    target_activities = record.get("targetActivities") or []
    if not target_activities:
        return [{**shared, **ACTIVITY_BLANK}]

    rows = []
    for ta in target_activities:
        row = {
            "target_species": flatten_name(ta.get("targetSpecies")),
            "activity_measure_group": flatten_name(ta.get("activityMeasureGroup")),
            "activity_measure_value": ta.get("activityMeasureValue"),
            "concentration": ta.get("concentration"),
            "unit": flatten_name(ta.get("unit")),
            "medium": flatten_name(ta.get("medium")),
            "cfu": ta.get("cfu"),
            "cfu_group": flatten_name(ta.get("cfuGroup")),
            "ph": ta.get("ph"),
            "ionic_strength": ta.get("ionicStrength"),
            "salt_type": ta.get("saltType"),
            "note": ta.get("note"),
            "reference": ta.get("reference"),
            "activity": ta.get("activity"),
        }
        rows.append({**shared, **row})
    return rows

# Step 5: explode one peptide's hemoliticCytotoxicActivities into row-dicts (dbaaspId + HC
# columns only, per the requested HC dataframe scope). One blank row if the list is empty.
HC_BLANK = {
    "target_cell": None, "activity_measure_for_lysis_group": None,
    "activity_measure_for_lysis_value": None, "concentration": None, "unit": None,
    "ph": None, "ionic_strength": None, "salt_type": None, "note": None,
    "reference": None, "activity": None,
}


def build_hc_rows(record):
    dbaasp_id = record.get("dbaaspId")
    hc_activities = record.get("hemoliticCytotoxicActivities") or []
    if not hc_activities:
        return [{"dbaaspId": dbaasp_id, **HC_BLANK}]

    rows = []
    for hc in hc_activities:
        row = {
            "dbaaspId": dbaasp_id,
            "target_cell": flatten_name(hc.get("targetCell")),
            "activity_measure_for_lysis_group": flatten_name(hc.get("activityMeasureForLysisGroup")),
            "activity_measure_for_lysis_value": hc.get("activityMeasureForLysisValue"),
            "concentration": hc.get("concentration"),
            "unit": flatten_name(hc.get("unit")),
            "ph": hc.get("ph"),
            "ionic_strength": hc.get("ionicStrength"),
            "salt_type": hc.get("saltType"),
            "note": hc.get("note"),
            "reference": hc.get("reference"),
            "activity": hc.get("activity"),
        }
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Flatten DBAASP detail JSONs into activity and hemolytic/cytotoxic CSVs."
    )
    parser.add_argument("--details_dir", type=Path, required=True,
                        help="Directory containing DBAASP detail *.json files")
    parser.add_argument("--out_activity_csv", type=Path, required=True,
                        help="Output path for the target-activity CSV")
    parser.add_argument("--out_hc_csv", type=Path, required=True,
                        help="Output path for the hemolytic/cytotoxic CSV")
    args = parser.parse_args()

    # Step 6: iterate every detail JSON file, building rows for both dataframes.
    detail_files = sorted(args.details_dir.glob("*.json"))
    print(f"{len(detail_files)} detail files found")

    activity_rows = []
    hc_rows = []
    for path in detail_files:
        record = json.loads(path.read_text())
        shared = build_peptide_shared_fields(record)
        activity_rows.extend(build_activity_rows(record, shared))
        hc_rows.extend(build_hc_rows(record))

    # Step 7: assemble the two dataframes.
    activity_df = pd.DataFrame(activity_rows)
    hc_df = pd.DataFrame(hc_rows)

    # Step 8: write both dataframes to disk.
    activity_df.to_csv(args.out_activity_csv, index=False)
    hc_df.to_csv(args.out_hc_csv, index=False)


if __name__ == "__main__":
    main()