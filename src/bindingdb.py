import pandas as pd
import re
import sqlite3

AFFINITY_MIN_NM = 0.0
AFFINITY_MAX_NM = 200.0

AFFINITY_COLS = [
    ("IC50 (nM)", "IC50"),
    ("Ki (nM)", "Ki"),
    ("Kd (nM)", "Kd"),
    ("EC50 (nM)", "EC50"),
]

LIGAND_ID_COLS = {
    "chembl": "ChEMBL ID of Ligand",
    "drugbank": "DrugBank ID of Ligand",
    "pubchem_cid": "PubChem CID",
    "pubchem_sid": "PubChem SID",
    "chebi": "ChEBI ID of Ligand",
    "kegg": "KEGG ID of Ligand",
    "zinc": "ZINC ID of Ligand",
    "bindingdb": "BindingDB MonomerID",
    "pdb_het": "Ligand HET ID in PDB",
}

def parse_affinity(val):
    if pd.isna(val):
        return None

    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip()
    if s in {"ND", "NA", "", "None"}:
        return None

    if "-" in s:
        try:
            lo, hi = map(float, s.split("-", 1))
            return (lo + hi) / 2
        except ValueError:
            return None

    s = re.sub(r"[<>~=]", "", s)

    try:
        return float(s)
    except ValueError:
        return None

def extract_affinity(row):
    for col, label in AFFINITY_COLS:
        val = parse_affinity(row.get(col))
        if val is not None:
            return val, label
    return None, None

def extract_ligand_ids(row):
    ids = {}
    for key, col in LIGAND_ID_COLS.items():
        val = row.get(col)
        if pd.notna(val):
            ids[key] = str(val)
    return ids

def get_ligands_from_bindingdb(uniprot):
    ligands = {}

    conn = sqlite3.connect("bindingdb/bindingdb.db")

    query = """
        SELECT b.*
        FROM bindingdb b
        JOIN bindingdb_targets t
          ON t.reactant_set_id = b."BindingDB Reactant_set_id"
        WHERE t.uniprot_id = ?
    """

    df = pd.read_sql_query(query, conn, params=(uniprot,))
    conn.close()

    if df.empty:
        return {"ligands": [], "ligand_count": 0}

    for _, row in df.iterrows():
        affinity, affinity_type = extract_affinity(row)
        if affinity is None:
            continue

        if not (AFFINITY_MIN_NM <= affinity <= AFFINITY_MAX_NM):
            continue

        smiles = row.get("Ligand SMILES")
        inchi_key = row.get("Ligand InChI Key")
        ligand_ids = extract_ligand_ids(row)

        dedup_key = (
            inchi_key
            or ligand_ids.get("chembl")
            or ligand_ids.get("bindingdb")
            or smiles
        )

        if not dedup_key:
            continue

        if (
            dedup_key not in ligands
            or affinity < ligands[dedup_key]["affinity"]
        ):
            ligands[dedup_key] = {
                "ligand_ids": ligand_ids,
                "smiles": smiles,
                "inchi_key": inchi_key,
                "affinity_type": affinity_type,
                "affinity": affinity,
            }

    return {
        "ligands": list(ligands.values()),
        "ligand_count": len(ligands),
    }