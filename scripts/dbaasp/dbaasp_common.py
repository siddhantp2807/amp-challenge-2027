"""
dbaasp_common.py
=================
Shared constants and helpers used by the DBAASP pipeline scripts:
    01_fetch_list_pages.py
    02_build_peptides_csv.py
    03_fetch_peptide_details.py

ENDPOINTS (confirmed empirically against the live API):

    GET https://dbaasp.org/peptides?limit=<N>&offset=<M>
        -> {"totalCount": <int>, "data": [
              {id, dbaaspId, name, nTerminus, sequence, sequenceLength,
               pdb, pubchemCid, cTerminus, complexity, synthesisType,
               monomers?: [...]},
              ...
           ]}
        `limit`/`offset` pagination is CONFIRMED working (offset=20 returns
        a different set of ids than offset=0 -- verified via curl).

    GET https://dbaasp.org/peptides/{dbaaspId}
        -> full peptide record, including physicoChemicalProperties,
           targetActivities, antibiofilmActivities,
           hemoliticCytotoxicActivities [sic, DBAASP's own spelling],
           synergies, sourceGenes, pdbs, smiles, articles, bonds, etc.
        NOTE: this endpoint takes the STRING dbaaspId (e.g. "DBAASPR_16"),
        NOT the numeric `id` field from the list endpoint above.

Both endpoints just need `Accept: application/json` -- no auth, no special
headers, plain GET (confirmed -- the earlier POST/session/CSRF-header
workarounds tried against the OLD query-string API at /api/v1 are not
needed for this one).
"""

import json
import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://dbaasp.org"
LIST_URL = f"{BASE_URL}/peptides"
DETAIL_URL_TMPL = f"{BASE_URL}/peptides/{{dbaasp_id}}"

HEADERS = {"Accept": "application/json"}

# Default directory / file layout across all pipeline steps. These are only
# DEFAULTS -- every script exposes a CLI flag to override its own paths
# (e.g. --list-pages-dir, --details-dir, --out, --checkpoint-file), so
# nothing here is actually hardcoded into the pipeline's behavior.
DEFAULT_RAW_DIR = Path("../../data/raw-data/dbaasp")
DEFAULT_LIST_PAGES_DIR = DEFAULT_RAW_DIR / "list_pages"       # step 1 output: one file per page
DEFAULT_DETAILS_DIR = DEFAULT_RAW_DIR / "details"             # step 3 output: one file per peptide
DEFAULT_DETAIL_CHECKPOINT_FILE = DEFAULT_RAW_DIR / "details_checkpoint.json"
DEFAULT_PEPTIDES_CSV = DEFAULT_RAW_DIR / "dbaasp-peptides-compiled.csv"       # step 2 output



def api_get(url, params=None, retries=5, timeout=30, backoff=2.0):
    """GET request with retry/backoff. Raises with the real response body on failure."""
    last_err, last_info = None, ""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            last_info = f"HTTP {r.status_code}, body[:300]={r.text[:300]!r}"
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            last_err = e
            wait = backoff ** attempt
            print(f"  [warn] {url} failed ({e}); retry {attempt}/{retries} in {wait:.0f}s",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} retries ({url}, params={params}): "
                        f"{last_err}\nLast response: {last_info}")


def flatten_value(v):
    """
    Turn a nested dict/list JSON value into a flat scalar/string suitable
    for a single wide-CSV cell.

    - {"name": "X"} or {"name": "X", "description": "Y"}  -> "X"
      (DBAASP uses this shape constantly -- synthesisType, complexity,
      targetSpecies, unit, medium, etc. are all small name(+description)
      lookup objects)
    - other dicts                                          -> compact JSON string
    - list of dicts with a "name" field                     -> "; "-joined names
    - list of scalars                                        -> "; "-joined values
    - None / scalar                                          -> unchanged

    This is a lossy summary by design -- the full raw JSON for every
    peptide/page stays cached on disk wherever each script's --*-dir
    argument points it, so nothing is actually discarded; this is just
    what goes in the flat CSV column.
    """
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, dict):
        if "name" in v and len(v) <= 2:
            return v["name"]
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        if not v:
            return None
        if all(isinstance(x, dict) for x in v):
            parts = [str(x.get("name", json.dumps(x, ensure_ascii=False))) for x in v]
            return "; ".join(parts)
        return "; ".join(str(x) for x in v)
    return str(v)