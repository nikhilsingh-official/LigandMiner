import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.table import Table
from rich.console import Console

console = Console()
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data/molecule/"


# ============================================================
# UNIVERSAL HELPERS
# ============================================================

def get_chembl_id(ligand: dict):
    if isinstance(ligand.get("ligand_ids"), dict):
        chembl = ligand["ligand_ids"].get("chembl")
        if chembl:
            return chembl

    ligand_id = ligand.get("ligand_id")
    if isinstance(ligand_id, str) and ligand_id.startswith("CHEMBL"):
        return ligand_id

    return None


# ============================================================
# PROBIS EXTRACTION
# ============================================================

def extract_ligands(pipeline_json):
    ligands = []

    for protein in pipeline_json.get("ranked_proteins", []):
        protein_id = protein.get("uniprot_id")
        probis_score = protein.get("probis_score")

        best_structure = protein.get("best_structure") or {}
        struct_id = best_structure.get("struct_id")

        for ligand in protein.get("ligands", []):
            affinity = ligand.get("affinity")
            if affinity is None:
                continue

            ligands.append({
                "protein_id": protein_id,
                "struct_id": struct_id,
                "ligand_id": ligand.get("ligand_id"),
                "ligand_ids": ligand.get("ligand_ids"),
                "smiles": ligand.get("smiles"),
                "affinity": affinity,
                "probis_score": probis_score
            })

    return ligands


# ============================================================
# CHEMBL METADATA FETCH (ENRICHED)
# ============================================================

def to_float_or_na(value):
    """
    Safely cast to float.
    Returns None if value is missing or not numeric.
    """
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

def fetch_chembl_metadata(chembl_id):
    try:
        url = f"{CHEMBL_BASE}{chembl_id}.json"
        r = requests.get(url, timeout=5)

        if r.status_code != 200:
            return chembl_id, None

        data = r.json()
        props = data.get("molecule_properties", {})

        return chembl_id, {
            "name": data.get("pref_name"),
            "type": data.get("molecule_type"),
            "phase": to_float_or_na(data.get("max_phase")),
            "first_approval": data.get("first_approval"),
            "mw": to_float_or_na(props.get("full_mwt")),
            "alogp": to_float_or_na(props.get("alogp")),
            "psa": to_float_or_na(props.get("psa")),
            "hbd": props.get("hbd"),
            "hba": props.get("hba"),
            "ro5": props.get("num_ro5_violations"),
        }

    except Exception:
        return chembl_id, None


def get_known_ligands_parallel(chembl_ids, max_workers=20):
    metadata_map = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_chembl_metadata, cid): cid
            for cid in chembl_ids
        }

        for i, future in enumerate(as_completed(futures), 1):
            cid, meta = future.result()

            if meta and meta["name"] is not None:
                metadata_map[cid] = meta

            if i % 25 == 0:
                console.print(f"[cyan]Checked {i}/{len(chembl_ids)}[/cyan]")

    return metadata_map


# ============================================================
# TABLE PRINTER
# ============================================================

def print_table(title, ligands):
    ligands_sorted = sorted(ligands, key=lambda x: x["affinity"])
    table = Table(title=title)

    columns = [
        "Rank","Protein","Structure","CHEMBL","Name","Phase",
        "IC50 nM","MW","ALogP","PSA","HBD","HBA","Ro5","Year","ProBis"
    ]

    for col in columns:
        table.add_column(col, justify="right" if col in ["Rank","IC50 nM","MW","ALogP","PSA","HBD","HBA","Ro5","Phase"] else "left")

    for i, lig in enumerate(ligands_sorted, start=1):
        m = lig["meta"]

        table.add_row(
            str(i),
            lig["protein_id"],
            lig["struct_id"],
            lig["chembl_id"],
            str(m["name"]),
            str(m["phase"]),
            f'{lig["affinity"]:.3g}',
            str(m["mw"]),
            str(m["alogp"]),
            str(m["psa"]),
            str(m["hbd"]),
            str(m["hba"]),
            str(m["ro5"]),
            str(m["first_approval"]),
            f'{lig["probis_score"]:.3f}' if lig["probis_score"] else "N/A"
        )

    console.print(table)


# ============================================================
# MAIN
# ============================================================

console.print("[bold yellow]Loading ProBis output...[/bold yellow]")

with open("output.json") as f:
    data = json.load(f)

ligands = extract_ligands(data)
console.print(f"[green]Extracted {len(ligands)} ligands[/green]")

chembl_ids = {get_chembl_id(l) for l in ligands if get_chembl_id(l)}
console.print(f"[yellow]Querying {len(chembl_ids)} CHEMBL IDs...[/yellow]")

metadata_map = get_known_ligands_parallel(chembl_ids)
console.print(f"[green]Fetched metadata for {len(metadata_map)} ligands[/green]")

fda_ligands, non_fda_ligands = [], []

for lig in ligands:
    cid = get_chembl_id(lig)
    if cid not in metadata_map:
        continue

    lig["chembl_id"] = cid
    lig["meta"] = metadata_map[cid]

    if metadata_map[cid]["phase"] == 4:
        fda_ligands.append(lig)
    else:
        non_fda_ligands.append(lig)

print_table("FDA APPROVED LIGANDS", fda_ligands)
print_table("NON-APPROVED LIGANDS", non_fda_ligands)
