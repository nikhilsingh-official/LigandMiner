import json
import re
import sqlite3
import sys
from collections import defaultdict
from enum import Enum
from typing import List
import pandas as pd
from Bio.PDB import PDBParser
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn
)
from pydantic import TypeAdapter

from src.pharma.pharmacophore.pharma3d.screening import Pharm3DScreening
from src.domain.pharm import Pharm2DConfig, Pharm3DConfig
from src.pharma.pharmacophore.pharma2d.screening import Pharm2DScreening
from src.domain.ligand import Ligand, LigandRecord
from src.domain.common import (
    MethodType,
    RESOLUTION_METHODS,
    NON_RESOLUTION_METHODS,
)
from src.domain.pipeline import (
    ProteinData,
    PipelineState,
)
from src.domain.scoring import (
    PDB_METHOD_SCORE,
    STRUCTURE_DB_MULTIPLIER,
    PDB_WEIGHTS,
    SWISS_MODEL_WEIGHTS,
    ALPHA_FOLD_WEIGHTS,
)
from src.domain.structure import (
    Chain,
    Pocket,
    ExperimentalStructure,
    AlphaFoldStructure,
    HomologyStructure,
    Structure,
)
from src.domain.uniprot import (
    UniProtEntry,
    SequenceAnnotations,
    Region
)

progress = Progress(
    SpinnerColumn(),
    TextColumn("{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeRemainingColumn(),
    TimeElapsedColumn()
)

import logging

logger = logging.getLogger("pharm2d")
logger.setLevel(logging.DEBUG)
logger.propagate = False

file_handler = logging.FileHandler("pharm2d.log")

file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

def fetch_interpro_json(id):
    url = f"https://www.ebi.ac.uk/interpro/api/entry/interpro/{id}/"
    res = session.get(url)
    if res.status_code != 200:
        print(f"Invalid response: {id}")
        return None
    return res.json()

def map_pdb_to_uniprot(resseq: int, mapping: dict) -> int | None:
    pdb_start = mapping["start"]["residue_number"]
    pdb_end = mapping["end"]["residue_number"]

    if not (pdb_start <= resseq <= pdb_end):
        return None

    return mapping["unp_start"] + (resseq - pdb_start)


def map_uniprot_chains_to_pdb(chains):
    pass


def fetch_uniprot_json_from_id(uniprot_id: str) -> UniProtEntry:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json?fields=accession,ft_domain,ft_act_site,ft_binding,uniparc_id,structure_3d,xref_alphafolddb,xref_smr,xref_interpro"
    res = session.get(url)
    res.raise_for_status()

    return UniProtEntry.model_validate(res.json())


def fetch_uniprot_json_from_interpro(
        interpro_id: str,
        progress: Progress,
        reviewed: bool = True,
        page_size: int = 500,
) -> List[UniProtEntry]:
    sub_task = progress.add_task(
        "      ↳ Fetching InterPro members' UniProt...",
        total=None
    )

    base_url = "https://rest.uniprot.org/uniprotkb/search"

    query = f"interpro-{interpro_id}"
    if reviewed:
        query = f"({query} AND reviewed:true)"

    params = {
        "query": query,
        "format": "json",
        "size": page_size,
    }

    results = []
    url = base_url
    page = 1

    while True:
        progress.update(
            sub_task,
            description=f"      ↳ Fetching InterPro members' UniProt (page {page})..."
        )

        res = session.get(url, params=params)
        if res.status_code != 200:
            raise RuntimeError(f"UniProt request failed: {res.status_code}")

        data = res.json()
        results.extend(data.get("results", []))

        progress.advance(sub_task)

        link_header = res.headers.get("Link", "")
        next_url = None

        for part in link_header.split(","):
            if 'rel="next"' in part:
                next_url = part[part.find("<") + 1: part.find(">")]
                break

        if not next_url:
            break

        url = next_url
        params = None
        page += 1

    task_obj = progress.tasks[sub_task]

    progress.update(
        sub_task,
        total=task_obj.completed,
        completed=task_obj.completed,
        description=f"[green]      ✔ Retrieved {len(results)} UniProt entries"
    )

    adapter = TypeAdapter(List[UniProtEntry])
    return adapter.validate_python(results)

def parse_uniprot_cross_references(
        uniprot_json: UniProtEntry,
) -> tuple[SequenceAnnotations, List[Structure]]:
    cross_refs = uniprot_json.uniProtKBCrossReferences or []

    annotations = SequenceAnnotations()
    structures: List[Structure] = []

    annotations.interpro_ids = [
        ref.id for ref in cross_refs if ref.database == "InterPro"
    ]

    VALID_DBS = {"PDB", "AlphaFoldDB", "SMR"}

    for ref in cross_refs:
        if ref.database not in VALID_DBS:
            continue

        if ref.database == "PDB":

            structure = ExperimentalStructure(
                struct_id=ref.id,
                database=ref.database,
                chains=[],
                cif=f"https://files.rcsb.org/download/{ref.id}.cif"
            )

            for prop in ref.properties or []:
                if prop.key == "Method":
                    structure.method = prop.value

                elif prop.key == "Resolution":
                    try:
                        structure.resolution = float(prop.value.replace(" A", ""))
                    except Exception:
                        structure.resolution = None

                elif prop.key == "Chains":
                    structure.chains = [Chain(**c) for c in parse_chains(prop.value)]

        elif ref.database == "SMR":
            structure = HomologyStructure(
                struct_id=ref.id,
                database=ref.database,
                chains=[],
            )

        elif ref.database == "AlphaFoldDB":
            structure = AlphaFoldStructure(
                struct_id=ref.id,
                database=ref.database,
                chains=[],
            )

        structures.append(structure)

    return annotations, structures


def parse_uniprot_features(
        uniprot_json: UniProtEntry,
        annotations: SequenceAnnotations,
) -> SequenceAnnotations:
    features = uniprot_json.features or []

    annotations.domains = [
        Region(start=f.location.start.value, end=f.location.end.value)
        for f in features
        if f.type == "Domain"
    ]

    annotations.active_sites = [
        Region(start=f.location.start.value, end=f.location.end.value)
        for f in features
        if f.type == "Active site"
    ]

    return annotations


def parse_uniprot_json(
        uniprot_json: UniProtEntry
) -> tuple[SequenceAnnotations, List[Structure]]:
    annotations, structures = parse_uniprot_cross_references(uniprot_json)
    annotations = parse_uniprot_features(uniprot_json, annotations)

    return annotations, structures


def parse_chains(chain_str):
    if not chain_str:
        return []

    results = []

    groups = [g.strip() for g in chain_str.split(",") if g.strip()]

    for group in groups:
        if "=" not in group:
            continue

        chain_part, residue_part = group.split("=", 1)

        chains = [
            c.strip()
            for c in chain_part.replace(",", "/").split("/")
            if c.strip()
        ]

        try:
            start, end = map(int, residue_part.split("-"))
        except ValueError:
            continue

        for chain in chains:
            results.append({
                "chain": chain,
                "start": start,
                "end": end
            })

    return results


from concurrent.futures import ThreadPoolExecutor, as_completed
import gemmi
from Bio.PDB import NeighborSearch
from Bio.PDB.Atom import Atom as BioAtom
import numpy as np
from rich.progress import Progress, TaskID


def clean_structure(input_cif, output_pdb, structure):
    input_path = Path(input_cif)
    output_path = Path(output_pdb)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selection = format_chains(structure)

    cmd = [
        "gemmi", "convert",
        str(input_path), str(output_path),
        "--select", selection,
        "--remove-lig-wat",
        "--remove-h",
        "--anisou=no",
        "--minimal",
        "--shorten",
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def build_neighbor_search(gemmi_structure):
    bio_atoms = []

    for model in gemmi_structure:
        for chain in model:
            for res in chain:
                for atom in res:
                    coord = atom.pos.tolist()
                    bio_atom = BioAtom(
                        atom.name, np.array(coord), atom.b_iso,
                        atom.occ, " ", atom.name, 0, element=atom.element.name.upper()
                    )
                    bio_atom.chain_id = chain.name
                    bio_atom.residue = res
                    bio_atoms.append(bio_atom)

    return NeighborSearch(bio_atoms), bio_atoms


def ligand_passes_basic_filters(residue, structure):
    comp_id = residue.name

    if comp_id in ARTIFACT_BLACKLIST:
        return False

    copies = sum(
        r.name == comp_id
        for m in structure
        for c in m
        for r in c
    )
    if copies >= 5:
        return False

    mw = sum(atom.element.weight for atom in residue)
    heavy_atoms = sum(atom.element.atomic_number > 1 for atom in residue)

    if mw < 200:
        return False
    if heavy_atoms < 10:
        return False

    occ = np.mean([a.occ for a in residue])
    if occ < 0.6:
        return False

    return True


def get_pocket_residues(ligand_residue, ns):
    CONTACT_DISTANCE = 5.0
    pocket = set()

    for atom in ligand_residue:
        neighbors = ns.search(atom.pos.tolist(), CONTACT_DISTANCE)

        for nb in neighbors:
            res = nb.residue

            if res.entity_type != gemmi.EntityType.Polymer:
                continue

            chain = nb.chain_id
            resseq = res.seqid.num
            icode = res.seqid.icode.strip() if res.seqid.icode else ""

            pocket.add((chain, resseq, icode))

    if len(pocket) < 6:
        return None

    return sorted(list(pocket))


ARTIFACT_BLACKLIST = {
    "SO4", "PO4", "HPO", "CIT", "ACT", "ACY", "FMT", "NO3", "CL", "BR", "IOD",
    "NA", "K", "MG", "CA", "ZN", "MN", "CU", "NI", "FE", "CO",
    "GOL", "EDO", "PEG", "PGE", "PG4", "PG5", "PG6", "MPD", "DMS", "IPA", "EOH",
    "AMS", "LIT", "TAR", "MAL", "SUC", "TRS", "HEP", "MES", "BIS", "BIC",
    "IMD", "TLA", "ACE", "EPE", "P6G", "P33", "PG0",
    "BOG", "DDM", "OG", "LDA", "SDS", "CHS",
    "DTT", "BME", "TCEP", "HOH"
}

class AminoAcid(Enum):
    ALA = "ALA"  # Alanine
    ARG = "ARG"  # Arginine
    ASN = "ASN"  # Asparagine
    ASP = "ASP"  # Aspartate
    CYS = "CYS"  # Cysteine
    GLN = "GLN"  # Glutamine
    GLU = "GLU"  # Glutamate
    GLY = "GLY"  # Glycine
    HIS = "HIS"  # Histidine
    ILE = "ILE"  # Isoleucine
    LEU = "LEU"  # Leucine
    LYS = "LYS"  # Lysine
    MET = "MET"  # Methionine
    PHE = "PHE"  # Phenylalanine
    PRO = "PRO"  # Proline
    SER = "SER"  # Serine
    THR = "THR"  # Threonine
    TRP = "TRP"  # Tryptophan
    TYR = "TYR"  # Tyrosine
    VAL = "VAL"  # Valine
    # non-standard
    MSE = "MSE"  # Selenomethionine
    HSD = "HSD"  # Histidine (neutral, delta-protonated)
    HSE = "HSE"  # Histidine (neutral, epsilon-protonated)
    HSP = "HSP"  # Histidine (positively charged)

    @classmethod
    def three_letter_from_one_letter(cls, code: str) -> AminoAcid:
        mapping: dict[str, AminoAcid] = {
            "A": cls.ALA, "R": cls.ARG, "N": cls.ASN, "D": cls.ASP,
            "C": cls.CYS, "Q": cls.GLN, "E": cls.GLU, "G": cls.GLY,
            "H": cls.HIS, "I": cls.ILE, "L": cls.LEU, "K": cls.LYS,
            "M": cls.MET, "F": cls.PHE, "P": cls.PRO, "S": cls.SER,
            "T": cls.THR, "W": cls.TRP, "Y": cls.TYR, "V": cls.VAL,
        }
        if code not in mapping:
            raise ValueError(f"Unknown one-letter code: {code}")
        return mapping[code]

    @property
    def one_letter_from_three_letter(self) -> str:
        mapping: dict[AminoAcid, str] = {
            AminoAcid.ALA: "A", AminoAcid.ARG: "R", AminoAcid.ASN: "N",
            AminoAcid.ASP: "D", AminoAcid.CYS: "C", AminoAcid.GLN: "Q",
            AminoAcid.GLU: "E", AminoAcid.GLY: "G", AminoAcid.HIS: "H",
            AminoAcid.ILE: "I", AminoAcid.LEU: "L", AminoAcid.LYS: "K",
            AminoAcid.MET: "M", AminoAcid.PHE: "F", AminoAcid.PRO: "P",
            AminoAcid.SER: "S", AminoAcid.THR: "T", AminoAcid.TRP: "W",
            AminoAcid.TYR: "Y", AminoAcid.VAL: "V",
        }
        return mapping.get(self, "X")  # X for non-standard

@dataclass
class StructureResidue(Residue):
    amino_acid: AminoAcid
    seq_id: int
    chain_id: str
    insertion_code: str

    @property
    def full_id(self):
        return f"{self.chain_id}:{self.seq_id}{self.insertion_code}:{self.amino_acid.value}"

    def __str__(self) -> str:
        return self.full_id

@dataclass
class StructureBindingSite:
    structure: Structure
    ligand: Ligand
    residues: List[PDBResidue]

def search_pubchem_by_smiles(smiles: str) -> dict | None:
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/JSON"
    response = session.get(url, params={"smiles": smiles})
    if response.status_code == 200:
        props = response.json()["PC_Compounds"][0]["props"]
        names = [p for p in props if p["urn"]["label"] == "IUPAC Name"]
        return names[0]["value"]["sval"] if names else None
    return None

def is_holo(input_cif, valid_chains: List[Chain]):
    structure = gemmi.read_structure(str(input_cif))

    ns, _ = build_neighbor_search(structure)

    valid_chain_ids = {c.chain if hasattr(c, "chain") else c for c in valid_chains}

    pocket_candidates = []

    for model in structure:
        for chain in model:
            for res in chain:
                if res.het_flag != "H":
                    continue

                if not ligand_passes_basic_filters(res, structure):
                    continue

                pocket = get_pocket_residues(res, ns)

                if not pocket:
                    continue

                filtered_pocket = [
                    (c, r, i) for (c, r, i) in pocket if c in valid_chain_ids
                ]

                if len(filtered_pocket) >= 6:
                    pocket_candidates.append(filtered_pocket)

    if not pocket_candidates:
        return None

    best_pocket = max(pocket_candidates, key=len)

    return best_pocket


def download_structure(structure: Structure, progress, sub_task, root):
    struct_id = structure.struct_id
    struct_id = struct_id.upper()

    progress.update(sub_task, description=f"   ↳ Downloading structures ({struct_id})...")

    structure.raw_structure_path.parent.mkdir(parents=True, exist_ok=True)

    if structure.raw_structure_path.exists():
        progress.advance(sub_task)
        return structure

    res = session.get(structure.cif)

    if res.status_code != 200:
        raise RuntimeError(
            f"Failed to download {struct_id} (HTTP {res.status_code})"
        )

    with open(structure.raw_structure_path, "wb") as f:
        f.write(res.content)

    progress.advance(sub_task)

    return structure


def download_structures(protein: ProteinData, progress: Progress, sub_task: TaskID, root="./", max_workers=6):
    downloaded = []

    root = Path(root)

    structures = protein.structures

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(download_structure, structure, progress, sub_task, root): structure
            for structure in structures
        }

        for future in as_completed(futures):
            structure = futures[future]
            struct_id = structure.struct_id
            try:
                result = future.result()
                if result is not None:
                    downloaded.append(result)
            except Exception as e:
                print(f"[DOWNLOAD FAIL] {struct_id}: {e}")

    return downloaded


from pathlib import Path
import shutil
import subprocess


def residues_to_fpocket_P(residues):
    parts = []

    for chain, resseq, icode in residues:
        icode = icode if icode else "-"
        parts.append(f"{resseq}:{icode}:{chain}")

    return ".".join(parts)


def run_fpocket(structure: Structure) -> Path:
    pdb_path = structure.structure_path
    desired_outdir = structure.fpocket_out

    work_dir = pdb_path.parent

    true_pocket = structure.true_pocket
    fpocket_p = residues_to_fpocket_P(true_pocket)

    subprocess.run(
        ["fpocket", "-f", pdb_path.name, "--custom-pocket", fpocket_p],
        cwd=work_dir,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    generated_outdir = work_dir / f"{pdb_path.stem}_out"

    if not generated_outdir.exists():
        raise RuntimeError("fpocket output directory was not created.")

    if desired_outdir.exists():
        shutil.rmtree(desired_outdir)

    shutil.move(str(generated_outdir), str(desired_outdir))

    return desired_outdir


from pathlib import Path

KEY_MAP = {
    "Score": "score",
    "Druggability Score": "druggability_score",
    "Number of Alpha Spheres": "number_of_alpha_spheres",
    "Total SASA": "total_sasa",
    "Polar SASA": "polar_sasa",
    "Apolar SASA": "apolar_sasa",
    "Volume": "volume",
    "Mean local hydrophobic density": "mean_local_hydrophobic_density",
    "Mean alpha sphere radius": "mean_alpha_sphere_radius",
    "Mean alp. sph. solvent access": "mean_alpha_sphere_solvent_access",
    "Apolar alpha sphere proportion": "apolar_alpha_sphere_proportion",
    "Hydrophobicity score": "hydrophobicity_score",
    "Volume score": "volume_score",
    "Polarity score": "polarity_score",
    "Charge score": "charge_score",
    "Proportion of polar atoms": "proportion_of_polar_atoms",
    "Alpha sphere density": "alpha_sphere_density",
    "Cent. of mass - Alpha Sphere max dist": "center_of_mass_alpha_sphere_max_dist",
    "Flexibility": "flexibility",
}


def parse_fpocket_info(fpocket_out: Path) -> list[Pocket]:
    info_files = list(fpocket_out.glob("*_info.txt"))
    if not info_files:
        raise RuntimeError(f"No fpocket info file in {fpocket_out}")

    info_file = info_files[0]

    pockets: list[Pocket] = []
    current: Pocket | None = None

    with open(info_file) as f:
        for line in f:
            line = line.strip()

            if line.startswith("Pocket"):
                if current:
                    pockets.append(current)

                pocket_id = int(line.split()[1])
                current = Pocket(pocket_id=pocket_id)

            elif ":" in line and current is not None:
                key, value = map(str.strip, line.split(":", 1))

                try:
                    value = float(value)
                except ValueError:
                    continue

                if key in KEY_MAP:
                    setattr(current, KEY_MAP[key], value)

        if current:
            pockets.append(current)

    return pockets


def score_alpha_fold_structure(structure: AlphaFoldStructure):
    raw = 0.0

    if structure.avg_local_score is not None:
        norm_plddt = structure.avg_local_score / 100.0
        raw += ALPHA_FOLD_WEIGHTS["avg_local_score"] * norm_plddt
    if structure.coverage is not None:
        raw += ALPHA_FOLD_WEIGHTS["coverage"] * structure.coverage

    calibrated = raw ** 1.25

    calibrated *= STRUCTURE_DB_MULTIPLIER["AlphaFoldDB"]

    structure.structural_plausibility = round(calibrated, 4)

    return structure


def score_alpha_fold_structures(structures: list[AlphaFoldStructure]):
    scored = [score_alpha_fold_structure(structure) for structure in structures]

    return scored


def score_pdb_structure(structure: ExperimentalStructure):
    resolution: float | None = structure.resolution
    method: MethodType = structure.method

    if method in RESOLUTION_METHODS and resolution is not None:

        resolution_norm = (1.0 / resolution) / (1.0 / 1.2)
        resolution_norm = max(0.0, min(1.0, resolution_norm))

        method_norm = PDB_METHOD_SCORE.get(method, 0.85)

        raw = (
                PDB_WEIGHTS["resolution"] * resolution_norm +
                PDB_WEIGHTS["method"] * method_norm
        )

    elif method in NON_RESOLUTION_METHODS and resolution is None:

        raw = PDB_METHOD_SCORE.get(method, 0.6)

    else:

        print(f"[WARNING] UNKNOWN METHOD: {method} has resolution {resolution}")
        raw = 0.4

    calibrated = raw ** 0.75
    calibrated *= STRUCTURE_DB_MULTIPLIER["PDB"]

    structure.structural_plausibility = round(calibrated, 4)
    return structure


def score_swiss_model_structure(structure: HomologyStructure):
    raw = 0.0
    raw += SWISS_MODEL_WEIGHTS["gmqe"] * structure.gmqe
    raw += SWISS_MODEL_WEIGHTS["domain_coverage"] * structure.domain_coverage
    raw += SWISS_MODEL_WEIGHTS["identity"] * structure.identity / 100
    raw += SWISS_MODEL_WEIGHTS["similarity"] * structure.similarity
    raw += SWISS_MODEL_WEIGHTS["avg_local_score"] * structure.avg_local_score

    if structure.qmean6_z is not None:
        qmean_norm = max(0.0, min(1.0, 1.0 + structure.qmean6_z / 6.0))
        raw += SWISS_MODEL_WEIGHTS["qmean6_z"] * qmean_norm

    calibrated = raw ** 1.05

    calibrated *= STRUCTURE_DB_MULTIPLIER["SMR"]

    structure.structural_plausibility = round(calibrated, 4)

    return structure


def score_swiss_model_structures(structures: list[HomologyStructure]):
    scored = [score_swiss_model_structure(structure) for structure in structures]

    return scored


def format_chains(structure):
    chain_ids = sorted(c.chain for c in structure.chains)
    selection = ",".join(f"{c}" for c in chain_ids)
    return selection


import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()

retry = Retry(
    total=5,
    connect=5,
    read=5,
    status=5,
    backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("HEAD", "GET", "OPTIONS"),
    raise_on_status=False,
)

adapter = HTTPAdapter(
    max_retries=retry,
    pool_connections=20,
    pool_maxsize=20,
)

session.mount("http://", adapter)
session.mount("https://", adapter)

session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)",
    "Accept": "*/*",
    "Connection": "keep-alive",
})


def pdb_residue_exists(chain, resseq):
    return (" ", resseq, " ") in chain


def has_sidechain(res):
    return any(
        atom.get_id() not in ("N", "CA", "C", "O")
        for atom in res.get_atoms()
    )


def normalize_resolution(res):
    res = min(max(res, 1.0), 3.5)
    return 1.0 - (res - 1.0) / (3.5 - 1.0)


def has_valid_binding_pocket(pocket_scores):
    return any(score > 0.0 for score in pocket_scores)


def extract_pocket_residues(pocket_cif):
    st = gemmi.read_structure(pocket_cif)
    residues = set()

    for model in st:
        for chain in model:
            for res in chain:
                # res.seqid.num is the residue number
                residues.add((chain.name, res.seqid.num))

    return sorted(residues)


def get_chains_from_motif(motif: str) -> str:
    chains = sorted(set(re.findall(r":([A-Za-z0-9])", motif)))

    if not chains:
        raise ValueError(f"No chains found in motif: {motif}")

    return ",".join(chains)


def expand_residues_spatially(
        pdb_path: str,
        seed_residues,
        radius: float = 8.0
):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)

    atoms = list(structure.get_atoms())
    ns = NeighborSearch(atoms)

    expanded = set(seed_residues)

    for chain_id, resseq, icode in seed_residues:
        for model in structure:
            if chain_id not in model:
                continue

            chain = model[chain_id]
            key = (" ", resseq, icode)

            if key not in chain:
                continue

            res = chain[key]

            for atom in res.get_atoms():
                for near_atom in ns.search(atom.coord, radius):
                    parent_res = near_atom.get_parent()
                    parent_chain = parent_res.get_parent().id
                    parent_resseq = parent_res.id[1]
                    parent_icode = parent_res.id[2]

                    expanded.add((parent_chain, parent_resseq, parent_icode))

    return list(expanded)


def residues_to_motif(structure):
    pockets = structure["pockets"]
    best_pocket = max(pockets, key=lambda p: p["binding_site_score"])
    residues = best_pocket["residues"]

    by_chain = {}
    for chain, res, icode in residues:
        by_chain.setdefault(chain, []).append(res)

    clauses = []
    for chain, reslist in by_chain.items():
        reslist = sorted(set(reslist))

        ranges = []
        start = prev = reslist[0]
        for r in reslist[1:]:
            if r == prev + 1:
                prev = r
            else:
                ranges.append((start, prev))
                start = prev = r
        ranges.append((start, prev))

        parts = [str(s) if s == e else f"{s}-{e}" for s, e in ranges]
        clauses.append(f":{chain} and ({','.join(parts)})")

    return "[" + " or ".join(clauses) + "]"


def probis_similarity_score(parsed):
    aligned = parsed["aligned_vertices"]
    sva = parsed["sva"]
    align_score = parsed["alignment_score"]

    sva_term = 1.0 / (1.0 + sva)

    base_score = (
            0.6 * aligned +
            0.25 * align_score +
            0.15 * sva_term
    )

    pocket = parsed.get("best_pocket_match")

    if pocket is None:
        return base_score * 0.5

    pocket_overlap = pocket["overlap_fraction"]
    alignment_in_pocket = pocket["alignment_in_pocket_fraction"]
    binding_score = pocket["binding_site_score"]

    pocket_term = (
            0.5 * pocket_overlap +
            0.3 * alignment_in_pocket +
            0.2 * binding_score
    )

    return base_score * (1.0 + pocket_term)


def parse_aligned_pair(pair_str):
    left, right = pair_str.split("=")

    _, t_resnum, t_chain = left.split(":")
    _, s_resnum, s_chain = right.split(":")

    return (
        (t_chain, int(t_resnum), " "),
        (s_chain, int(s_resnum), " "),
    )


def parse_results(target: Structure, similar: Structure, progress: Progress, indent=3):
    main_task = progress.add_task(
        f"{" " * indent}↳ Parsing results for {similar.struct_id}...",
        total=1
    )

    with open(similar.probis_results) as f:
        results = json.load(f)

    if not results:
        print("[SKIP] Not enough similarity, results.json is empty.")
        return None

    alignments = results[0]["alignment"]

    best = max(
        alignments,
        key=lambda a: a["scores"]["aligned_vertices"]
    )

    scores = best["scores"]

    aligned_target = set()
    aligned_similar = set()

    for entry in best["aligned_residues"]:
        t_res, s_res = parse_aligned_pair(entry["a"])
        aligned_target.add(t_res)
        aligned_similar.add(s_res)

    pocket_matches = []

    for pocket in target.fpockets:
        pocket_res = set(pocket.residues)
        if not pocket_res:
            continue

        overlap = aligned_target & pocket_res

        if not overlap:
            continue

        overlap_fraction = len(overlap) / len(pocket_res)
        coverage = len(overlap) / len(aligned_target)

        post_score = (
                0.45 * overlap_fraction +
                0.35 * coverage +
                0.20 * pocket.binding_site_score
        )

        pocket_matches.append({
            "pocket_id": pocket.pocket_id,
            "post_score": post_score,
            "overlap_count": len(overlap),
            "overlap_fraction": overlap_fraction,
            "alignment_in_pocket_fraction": coverage,
            "binding_site_score": pocket.binding_site_score,
        })

    best_pocket_match = (
        max(pocket_matches, key=lambda x: x["post_score"])
        if pocket_matches else None
    )

    progress.update(
        main_task,
        f"{" " * indent}✔ Parsed results for {similar.struct_id}"
    )
    progress.advance(main_task)

    return {
        "aligned_vertices": scores["aligned_vertices"],
        "sva": scores["sva"],
        "rmsd": scores["rmsd"],
        "alignment_score": scores["alignment_score"],
        "z_score": scores["z_score"],
        "aligned_target_residues": aligned_target,
        "aligned_similar_residues": aligned_similar,
        "best_pocket_match": best_pocket_match,
        "all_pocket_matches": pocket_matches,
    }


def chembl_targets_from_uniprot(uniprot):
    url = "https://www.ebi.ac.uk/chembl/api/data/target/search.json"
    params = {"q": uniprot}

    res = session.get(url, params=params, timeout=30)
    if not res.ok:
        return []

    data = res.json()

    targets = []
    for t in data.get("targets", []):
        for comp in t.get("target_components", []):
            if comp.get("accession") == uniprot:
                targets.append(t["target_chembl_id"])
                break

    return targets


VALID_TYPES = {"IC50", "Ki", "Kd", "EC50"}


def chembl_activities_from_targets(target_ids, limit=200):
    activities = []

    for tid in target_ids:
        url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
        params = {
            "target_chembl_id": tid,
            "standard_type__in": ",".join(VALID_TYPES),
            "standard_relation": "=",
            "limit": limit
        }

        res = session.get(url, params=params, timeout=30)
        if not res.ok:
            continue

        activities.extend(res.json().get("activities", []))

    return activities


def parse_chembl_activities(activities):
    ligands = {}

    for act in activities:
        try:
            value = float(act["standard_value"])
            ligand_id = act["molecule_chembl_id"]
            affinity_type = act["standard_type"]
            smiles = act["canonical_smiles"]

            if not smiles:
                continue

        except (KeyError, TypeError, ValueError):
            continue

        if ligand_id not in ligands or value < ligands[ligand_id]["affinity"]:
            ligands[ligand_id] = {
                "ligand_id": ligand_id,
                "smiles": smiles,
                "affinity_type": affinity_type,
                "affinity": value
            }

    return {
        "ligands": list(ligands.values()),
        "ligand_count": len(ligands)
    }


def get_ligands_from_chembl(uniprot: str):
    targets = chembl_targets_from_uniprot(uniprot)
    if not targets:
        return None

    activities = chembl_activities_from_targets(targets)
    if not activities:
        return None

    return parse_chembl_activities(activities)


def get_swiss_model_intermediate(swiss_model_id: str):
    swiss_model_id = swiss_model_id.upper()

    intermediate_url = f"https://swissmodel.expasy.org/repository/uniprot/{swiss_model_id}.json"
    res = session.get(intermediate_url)

    if res.status_code != 200:
        raise RuntimeError(
            f"Failed to download {swiss_model_id} (HTTP {res.status_code})"
        )

    json = res.json()
    return json


def get_alpha_fold_intermediate(alphafold_id: str):
    alphafold_id = alphafold_id.upper()

    intermediate_url = f"https://alphafold.ebi.ac.uk/api/uniprot/summary/{alphafold_id}.json"
    res = session.get(intermediate_url)

    if res.status_code != 200:
        raise RuntimeError(
            f"Failed to download {alphafold_id} (HTTP {res.status_code})"
        )

    json = res.json()
    return json


def parse_swiss_model_intermediate_json(json_data, uniprot: str):
    parsed_structures: list[HomologyStructure] = []

    for idx, structure in enumerate(json_data.get("result", {}).get("structures", [])):
        if structure.get("method") != "HOMOLOGY MODELLING":
            continue

        chains = [
            Chain(
                chain=chain["id"],
                start=structure["from"],
                end=structure["to"]
            )
            for chain in structure.get("chains", [])
        ]

        parsed_structures.append(
            HomologyStructure(
                struct_id=f"SM-{uniprot}-{idx}",
                database="SMR",
                cif=structure.get("modelcif"),
                chains=chains,

                coverage=structure.get("coverage"),
                gmqe=structure.get("gmqe"),
                identity=structure.get("identity"),
                similarity=structure.get("similarity"),

                avg_local_score=structure.get("qmean", {}).get("avg_local_score"),
                qmean6_z=structure.get("qmean", {}).get("qmean6_z_score"),

                provider=structure.get("provider"),
                template=structure.get("template"),
                created_date=structure.get("created_date"),
                method="HOMOLOGY MODELLING"
            )
        )

    return parsed_structures


def parse_alpha_fold_intermediate_json(json_data, uniprot):
    parsed_structures: list[AlphaFoldStructure] = []

    for idx, entry in enumerate(json_data.get("structures", [])):
        summary = entry.get("summary", {})

        if summary.get("provider") != "AlphaFold DB":
            continue

        chains = []
        for entity in summary.get("entities", []):
            for chain_id in entity.get("chain_ids", []):
                chains.append(
                    Chain(
                        chain=chain_id,
                        start=summary.get("uniprot_start"),
                        end=summary.get("uniprot_end")
                    )
                )

        parsed_structures.append(
            AlphaFoldStructure(
                struct_id=f"AF-{uniprot}-{idx}",
                database="AlphaFoldDB",
                cif=summary.get("model_url"),
                chains=chains,

                coverage=summary.get("coverage"),
                confidence_type=summary.get("confidence_type"),
                avg_local_score=summary.get("confidence_avg_local_score"),

                provider=summary.get("provider"),
                created_date=summary.get("created")
            )
        )

    return parsed_structures


def structure_selection_score(structure: Structure, w_structure=0.7, w_pocket=0.3):
    structural = structure.structural_plausibility

    best_pocket = max(
        (p.binding_site_score
         for p in structure.fpockets),
        default=0.0
    )

    if structural <= 0.0 or best_pocket <= 0.0:
        return 0.0

    return (structural ** w_structure) * (best_pocket ** w_pocket) * structure.true_pocket_certainty


STRUCTURE_TYPE_MULTIPLIER = {
    "PDB": 1.0,
    "SMR": 0.8,
    "AlphaFoldDB": 0.6,
}

PDB_RAW_WEIGHTS = {
    "resolution": 0.70,
    "method": 0.30,
}

PDB_METHOD_SCORE = {
    "X-ray": 1.0,
    "EM": 0.8,
    "Neutron": 0.7,
    "NMR": 0.6
}


def feature_structure_enrichment(structure: Structure, domain: List[Region], active_sites: List[Region]):
    chains = structure.chains

    #TODO: Add total coverage as tiebreaker
    #TODO: Add recency as tiebreaker
    #TODO: Fix the design where resolution < 1.0 is treated as 1.0, causing structures with 0.85 and 0.98 to get same score

    if not chains:
        return structure

    if domain:
        max_domain = max(domain, key=lambda d: d.end - d.start + 1)

        domain_len = max_domain.end - max_domain.start + 1
        domain_coverages = []

        for chain in chains:
            overlap_start = max(chain.start, max_domain.start)
            overlap_end = min(chain.end, max_domain.end)

            if overlap_start <= overlap_end:
                overlap_len = overlap_end - overlap_start + 1
                domain_coverages.append(overlap_len / domain_len)

        structure.domain_coverage = max(domain_coverages) if domain_coverages else 0.0
    else:
        structure.domain_coverage = 0.0

    total_sites = len(active_sites)
    covered_sites = 0

    for site in active_sites:
        site_present = False

        for chain in chains:
            if chain.start <= site.start and chain.end >= site.end:
                site_present = True
                break

        if site_present:
            covered_sites += 1

    structure.active_site_coverage = (
        covered_sites / total_sites if total_sites > 0 else 0.0
    )

    return structure


def find_best_swiss_model_structure(structure: Structure, uniprot: str):
    intermediate_json = get_swiss_model_intermediate(structure.struct_id)
    structures = parse_swiss_model_intermediate_json(intermediate_json, uniprot)
    scored_structures = score_swiss_model_structures(structures)
    best_struct = max(scored_structures, key=lambda s: s.structural_plausibility)
    return best_struct


def find_best_alpha_fold_structure(structure: Structure, uniprot: str):
    intermediate_json = get_alpha_fold_intermediate(structure.struct_id)
    structures = parse_alpha_fold_intermediate_json(intermediate_json, uniprot)
    scored_structures = score_alpha_fold_structures(structures)
    best_struct = max(scored_structures, key=lambda s: s.structural_plausibility)
    return best_struct


STRUCTURES_DIR = Path("structures")
COMPARISON_DIR = Path("probis_compare")


def add_paths(uniprot: str, structure: Structure):
    struct_id = structure.struct_id

    base = STRUCTURES_DIR / uniprot / struct_id

    structure.raw_structure_path = base / f"{struct_id}_raw.cif"
    structure.structure_path = base / f"{struct_id}.pdb"

    structure.fpocket_out = base / "fpocket_out"
    structure.probis_out = base / "probis_out"
    structure.prank_out = base / "prank_out"

    structure.fpocket_pockets = structure.fpocket_out / "pockets"
    structure.fpocket_txt = structure.fpocket_out / f"{struct_id}_info.txt"

    structure.probis_srf = structure.probis_out / f"{struct_id}.srf"

    structure.probis_compare = COMPARISON_DIR / uniprot / struct_id
    structure.probis_nosql = structure.probis_compare / "probis.nosql"
    structure.probis_results = structure.probis_compare / "results.json"

    structure.prank_csv = structure.prank_out / f"{struct_id}.pdb_predictions.csv"

    return structure


def metadata_structure_enrichment(structure: Structure, domain: List[Region], active_sites: List[Region], uniprot: str):
    if structure.database == "SMR":
        structure = find_best_swiss_model_structure(structure, uniprot)
    elif structure.database == "AlphaFoldDB":
        structure = find_best_alpha_fold_structure(structure, uniprot)
    else:
        structure = score_pdb_structure(structure)
    structure = feature_structure_enrichment(structure, domain, active_sites)
    structure = add_paths(uniprot, structure)
    return structure


def metadata_structures_enrichment(structures: list[Structure], domain: List[Region], active_sites: List[Region],
                                   uniprot: str):
    return [metadata_structure_enrichment(s, domain, active_sites, uniprot) for s in structures]


def pdb_specific_enrichment(structure):
    struct_id = structure["struct_id"]
    structure["cif"] = f"https://files.rcsb.org/download/{struct_id}.cif"
    return structure


def pdb_specific_filtration(structure):
    method = structure.get("method")
    resolution = structure.get("resolution")
    if method not in {"X-ray", "EM", "Neutron", "NMR"}:
        return False
    elif not resolution is None and resolution > 3.0:
        return False
    return True


def fpocket_enrichment(data):
    structures = data.structures
    for structure in structures:
        struct_id = structure.struct_id
        struct_id = struct_id.upper()

        if not structure.structure_path.exists():
            print(f"[FPOCKET SKIP] Missing PDB for {struct_id}")
            structure.fpockets = []
            structure.pocket_num = 0
            continue

        if structure.fpockets:
            continue

        if structure.fpocket_txt.exists() and structure.fpocket_txt.stat().st_size > 0:
            try:
                structure.fpockets = parse_fpocket_info(structure.fpocket_out)
                structure.pocket_num = len(structure.fpockets)
                continue
            except Exception as e:
                print(f"[FPOCKET CACHE PARSE FAIL] {struct_id}: {e}")
                structure.fpockets = []
                structure.pocket_num = 0

        if structure.fpocket_out.exists():
            shutil.rmtree(structure.fpocket_out)

        try:
            run_fpocket(structure)
            structure.fpockets = parse_fpocket_info(structure.fpocket_out)
            structure.pocket_num = len(structure.fpockets)
        except Exception as e:
            print(f"[FPOCKET RUN FAIL] {struct_id}: {e}")
            structure.fpockets = []
            structure.pocket_num = 0

    return data


def parse_fpocket_atm_pdb(pdb_path):
    st = gemmi.read_structure(str(pdb_path))

    residues = set()

    for model in st:
        for chain in model:
            for residue in chain:
                if residue.name == '':
                    continue

                resseq = residue.seqid.num
                icode = residue.seqid.icode
                chain_id = chain.name

                residues.add((chain_id, resseq, icode))

    return sorted(residues)


BINDING_SITE_WEIGHTS = {
    "domain_coverage": 0.45,
    "active_site_factor": 0.30,
    "geometry": 0.15,
    "physical_sanity": 0.10,
}


def geometry_score(pocket):
    volume = min(pocket.get("Volume", 0.0) / 400.0, 1.0)
    alpha_density = min(pocket.get("Alpha sphere density", 0.0) / 3.0, 1.0)
    volume_score = min(pocket.get("Volume score", 0.0) / 4.0, 1.0)
    return (volume + alpha_density + volume_score) / 3.0


def physical_sanity_score(pocket):
    rigidity = max(1.0 - pocket.get("Flexibility", 1.0), 0.0)

    solvent_access = pocket.get("Mean alp. sph. solvent access", 1.0)
    burial = max(1.0 - solvent_access, 0.0)

    polar_frac = pocket.get("Proportion of polar atoms", 0.0) / 100.0
    polarity_ok = max(1.0 - abs(polar_frac - 0.5), 0.0)

    return 0.4 * rigidity + 0.3 * burial + 0.3 * polarity_ok


def compute_binding_site_score(pocket):
    active_site = pocket.get("active_site_coverage", 0.0)
    purity = pocket.get("pocket_domain_purity", 0.0)

    geo = geometry_score(pocket)
    phys = physical_sanity_score(pocket)

    score = (
            geo
            * phys
            * (1.0 + 2.5 * active_site)
            * (0.5 + purity)
    )

    return round(score, 4)


def score_pockets(
        protein: ProteinData,
        *,
        residue_sasa=None,
        min_surface_residues=6,
        min_surface_sasa=5.0,
):
    domains = protein.annotations.domains if protein.annotations else []
    active_sites = protein.annotations.active_sites if protein.annotations else []
    structures = protein.structures

    main_domain = max(domains, key=lambda d: d.end - d.start, default=None)
    domain_residues = (
        set(range(main_domain.start, main_domain.end + 1))
        if main_domain else set()
    )

    active_site_residues = {
        r
        for site in active_sites
        for r in range(site.start, site.end + 1)
    }

    for structure in structures:
        for pocket in structure.fpockets:
            residues = pocket.uniprot_indices

            if not residues or not domain_residues:
                continue

            pocket_res = {r[1] for r in residues}

            if residue_sasa is not None:
                surface_res = {
                    r for r in pocket_res
                    if residue_sasa.get(r, 0.0) >= min_surface_sasa
                }
            else:
                surface_res = pocket_res

            if len(surface_res) < min_surface_residues:
                continue

            surface_fraction = len(surface_res) / len(pocket_res)

            hit_domain = pocket_res & domain_residues
            hit_active = pocket_res & active_site_residues

            pocket.domain_coverage = len(hit_domain) / len(domain_residues)
            pocket.pocket_domain_purity = len(hit_domain) / len(pocket_res)
            pocket.active_site_coverage = (
                len(hit_active) / len(active_site_residues)
                if active_site_residues else 0.0
            )

            druggability = pocket.druggability_score

            pocket.binding_site_score = (
                    0.30 * pocket.pocket_domain_purity
                    + 0.25 * pocket.active_site_coverage
                    + 0.25 * druggability
                    + 0.20 * surface_fraction
            )

    return protein


def pocket_residue_enrichment(protein: ProteinData):
    for structure in protein.structures:

        for pocket in structure.fpockets:
            path = structure.fpocket_pockets / f"pocket{pocket.pocket_id}_atm.pdb"
            pocket.residues = parse_fpocket_atm_pdb(path)

    return map_residues_to_uniprot(protein)


def map_residues_to_uniprot(protein: ProteinData):
    for structure in protein.structures:

        use_sifts = isinstance(structure, ExperimentalStructure)
        sifts = structure.sifts_mapping if use_sifts else None

        for pocket in structure.fpockets:
            mapped_residues = []

            for chain, resseq, icode in pocket.residues:

                if use_sifts and sifts and chain == sifts["chain_id"]:
                    unp_res = map_pdb_to_uniprot(resseq, sifts)

                    if unp_res is not None:
                        mapped_residues.append((chain, unp_res, ""))
                        continue

                try:
                    chain_info = next(c for c in structure.chains if c.chain == chain)
                except StopIteration:
                    print("CHAIN MAPPING FAILURE")
                    print(f"Protein: {protein.uniprot_id}")
                    print(f"Structure ID: {structure.struct_id}")
                    print(f"Structure type: {type(structure).__name__}")
                    print(f"Missing chain: {chain}")
                    print(f"Pocket residues entry: {(chain, resseq, icode)}")

                    print("Available chains in structure:")
                    for c in structure.chains:
                        print(f"  chain={c.chain}, start={getattr(c, 'start', None)}")

                    raise

                unp_res = resseq - 1 + chain_info.start
                mapped_residues.append((chain, unp_res, icode))

            pocket.uniprot_indices = mapped_residues

    return protein


def merge_ligand_outputs(*outputs):
    merged = {}

    for out in outputs:
        if not out:
            continue

        for lig in out.get("ligands", []):
            dedup_key = (
                    lig.get("inchi_key")
                    or lig.get("ligand_id")
                    or lig.get("smiles")
            )

            if not dedup_key:
                continue

            if (
                    dedup_key not in merged
                    or lig["affinity"] < merged[dedup_key]["affinity"]
            ):
                merged[dedup_key] = lig

    return {
        "ligands": list(merged.values()),
        "ligand_count": len(merged),
    }


def get_pocket_from_prank(structure: Structure):
    subprocess.run(
        [
            "prank",
            "predict",
            "-f", str(structure.structure_path),
            "-o", str(structure.prank_out)
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    df = pd.read_csv(structure.prank_csv, skipinitialspace=True)

    if df.empty:
        return None, None

    best_row = df.loc[df["probability"].idxmax()]

    probability = float(best_row["probability"])

    residue_ids_str = best_row["residue_ids"]

    residues = []

    for token in residue_ids_str.split():
        chain, resseq_str = token.split("_")

        resseq = int(''.join(c for c in resseq_str if c.isdigit()))
        icode = ''.join(c for c in resseq_str if c.isalpha())

        residues.append((chain, resseq, icode))

    return probability, residues

def get_uniparc_json(uniparc_id: str):
    url = f"https://www.uniprot.org/uniparc/{uniparc_id}.json"
    res = session.get(url, timeout=30)

    if not res.ok:
        raise RuntimeError(f"Failed to fetch UniParc data for {uniparc_id} (HTTP {res.status_code})")

    return res.json()

def parse_uniparc_json(json_data) -> List[str]:
    return [entry["id"] for entry in json_data.get("uniParcCrossReferences", []) if entry.get("database").startswith("UniProtKB")]

def parse_sifts_json(mapping_json, structure: ExperimentalStructure, accessions: List[str]):
    pdb_id = structure.struct_id.lower()

    if pdb_id not in mapping_json:
        raise ValueError(f"No SIFTS entry for {structure.struct_id}")

    entry = mapping_json[pdb_id].get("UniProt", {})

    struct_chains = {c.chain for c in structure.chains}

    for accession in accessions:
        if accession not in entry:
            continue

        mappings = entry[accession].get("mappings", [])

        for m in mappings:
            label_chain = m.get("struct_asym_id")  # mmCIF label chain
            auth_chain = m.get("chain_id")        # author chain

            if label_chain in struct_chains or auth_chain in struct_chains:
                return m

    return None

def prepare_protein_metadata(
    protein: ProteinData,
    entry: UniProtEntry,
    progress: Progress | None = None,
    indent: int = 3
) -> None:
    progress = progress or NullProgress()

    task = progress.add_task(
        f"{' ' * indent}↳ Parsing UniProt for {protein.uniprot_id}...",
        total=1
    )

    annotations, structures = parse_uniprot_json(entry)
    protein.annotations = annotations

    accessions = list(dict.fromkeys(
        [protein.uniprot_id] + (protein.other_accessions or [])
    ))

    for structure in structures:
        if not isinstance(structure, ExperimentalStructure):
            continue

        mapping = fetch_sifts_json(structure.struct_id)

        try:
            structure.sifts_mapping = parse_sifts_json(
                mapping,
                structure,
                accessions
            )
        except ValueError:
            structure.sifts_mapping = None

    protein.structures = structures

    progress.advance(task)
    progress.update(
        task,
        description=f"[green]{' ' * indent}✔ UniProt parsed for {protein.uniprot_id}"
    )


def manage_interpro(interpro_ids: list[str], progress: Progress | None = None):
    progress = progress or NullProgress()

    main_task = progress.add_task(
        f"   ↳ Searching for candidate similar proteins to {uniprot}...",
        total=2
    )

    progress.update(
        main_task,
        description=f"   ↳ Searching for candidate similar proteins to {uniprot} (fetching InterPro family)…"
    )

    least_abstracted = fetch_least_abstracted_interpro_id(interpro_ids, progress)

    progress.update(
        main_task,
        description=f"   ↳ Searching for candidate similar proteins to {uniprot} (fetching proteins' UniProt entries)…"
    )

    progress.advance(main_task)

    results: List[UniProtEntry] = fetch_uniprot_json_from_interpro(least_abstracted["id"], progress)

    progress.update(main_task, description=f"   ✔ Found {len(results)} candidate similar proteins to {uniprot}…)")

    progress.advance(main_task)

    return results


def fetch_least_abstracted_interpro_id(interpro_ids, progress: Progress):
    sub_task = progress.add_task("      ↳ Searching for InterPro group...", total=len(interpro_ids))

    least_abstracted = None

    for interpro_id in interpro_ids:
        interpro_json = fetch_interpro_json(interpro_id)

        if interpro_json:
            t = interpro_json["metadata"]["type"]

            if t == "family":
                remaining = len(interpro_ids) - progress.tasks[sub_task].completed + 1
                progress.update(sub_task, description=f"      ✔ Found InterPro family: {interpro_id}")
                progress.advance(sub_task, remaining)
                return {"id": interpro_id, "type": "family"}

            if t == "domain" and least_abstracted is None:
                least_abstracted = {"id": interpro_id, "type": "domain"}

        progress.advance(sub_task)

    progress.update(sub_task,
                    description=f"      ✔ Found InterPro {least_abstracted['type']}: {least_abstracted['id']}")
    return least_abstracted


class NullProgress:
    def add_task(self, *args, **kwargs):
        return None

    def update(self, *args, **kwargs):
        pass

    def advance(self, *args, **kwargs):
        pass

    def remove_task(self, *args, **kwargs):
        pass


def shortlist_structures(structures, top_n=100):
    structures.sort(
        key=lambda s: s.structural_plausibility or 0.0,
        reverse=True
    )
    return structures[:top_n]

def select_best_structure(protein: ProteinData, progress: Progress | None = None, indent=3):
    progress = progress or NullProgress()

    main_task = progress.add_task(
        f"{" " * indent}↳ Selecting best structure for {protein.uniprot_id}...",
        total=5
    )

    protein.structures = metadata_structures_enrichment(protein.structures, protein.annotations.domains,
                                                        protein.annotations.active_sites, protein.uniprot_id)

    protein.structures = shortlist_structures(protein.structures, top_n=100)

    progress.update(main_task,
                    description=f"{" " * indent}↳ Selecting best structure for {protein.uniprot_id} (downloading structures)…")

    sub_task = progress.add_task(f"{" " * indent * 2}↳ Downloading structures...", total=len(protein.structures))

    download_structures(protein, progress, sub_task, "structures")

    progress.update(sub_task, description=f"[green]{" " * indent * 2}✔ Downloaded structures")
    progress.advance(main_task)

    progress.update(main_task,
                    description=f"{" " * indent}↳ Selecting best structure for {protein.uniprot_id} (preparing structures)…")
    sub_task = progress.add_task(f"{" " * indent * 2}↳ Preparing structures...", total=len(protein.structures))
    for structure in protein.structures:
        progress.update(sub_task, description=f"{" " * indent * 2}↳ Preparing structures ({structure.struct_id})...")
        clean_structure(structure.raw_structure_path, structure.structure_path, structure)
        structure.true_pocket = is_holo(structure.raw_structure_path, structure.chains)
        if structure.true_pocket:
            structure.true_pocket_certainty = 1.0
        else:
            probability, residues = get_pocket_from_prank(structure)
            structure.true_pocket = residues
            structure.true_pocket_certainty = probability
        progress.advance(sub_task)

    progress.update(sub_task, description=f"[green]{" " * indent * 2}✔ Prepared structures")
    progress.advance(main_task)

    progress.update(main_task,
                    description=f"{" " * indent}↳ Selecting best structure for {protein.uniprot_id} (fpocket enrichment)…")
    sub_task = progress.add_task(f"{" " * indent * 2}↳ Running fpocket...", total=1)
    protein = fpocket_enrichment(protein)

    progress.advance(sub_task)
    progress.update(sub_task, description=f"[green]{" " * indent * 2}✔ Enriched structures by fpocket")
    progress.advance(main_task)

    progress.update(main_task,
                    description=f"{" " * indent}↳ Selecting best structure for {protein.uniprot_id} (residue enrichment)…")
    sub_task = progress.add_task(f"{" " * indent * 2}↳ Enriching pocket residues...", total=1)

    protein = pocket_residue_enrichment(protein)

    progress.advance(sub_task)
    progress.update(sub_task, description=f"[green]{" " * indent * 2}✔ Enriched pocket residues")
    progress.advance(main_task)

    progress.update(main_task,
                    description=f"{" " * indent}↳ Selecting best structure for {protein.uniprot_id} (scoring)…")
    sub_task = progress.add_task(f"{" " * indent * 2}↳ Scoring pockets...", total=1)
    protein = score_pockets(protein)
    best_structure = max(
        protein.structures,
        key=lambda s: structure_selection_score(
            s,
            w_structure=0.7,
            w_pocket=0.3
        )
    )

    progress.advance(sub_task)
    progress.update(sub_task, description=f"[green]{" " * indent * 2}✔ Structures scored")
    progress.advance(main_task)

    progress.update(main_task,
                    description=f"{" " * indent}↳ ✔ Best structure selected for {protein.uniprot_id}: {best_structure.struct_id}")

    return best_structure


def manage_results(target_structure: Structure, similar_structure: Structure, progress: Progress | None = None,
                   indent=3):
    progress = progress or NullProgress()

    get_results(target_structure, similar_structure, progress, indent=indent * 2)

    parsed = parse_results(target_structure, similar_structure, progress, indent=indent * 2)

    return parsed


def surface_extraction(structure: Structure, progress: Progress | None = None, indent=3):
    progress = progress or NullProgress()

    chains = sorted(c.chain for c in structure.chains)
    chains_selection = ",".join(f"{c}" for c in chains)

    Path(structure.probis_out).mkdir(parents=True, exist_ok=True)

    if structure.probis_srf and structure.probis_srf.exists() and structure.probis_srf.stat().st_size > 0:
        return structure.probis_srf

    if not structure.structure_path.exists():
        raise FileNotFoundError(structure.structure_path)

    cmd = [
        "./probis",
        "-extract",
        "-f1", str(structure.structure_path),
        "-c1", chains_selection,
        "-srffile", str(structure.probis_srf)
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def compare_surfaces(
        target_structure: Structure,
        similar_structure: Structure,
        progress: Progress | None = None,
        indent=3
):
    progress = progress or NullProgress()

    main_task = progress.add_task(
        f"{" " * indent}↳ Comparing {target_structure.struct_id} with {similar_structure.struct_id}...",
        total=1
    )

    if not target_structure.probis_srf.exists():
        raise FileNotFoundError(target_structure.probis_srf)
    if not similar_structure.probis_srf.exists():
        raise FileNotFoundError(similar_structure.probis_srf)

    target_chains = sorted(c.chain for c in target_structure.chains)
    c1_selection = ",".join(f"{c}" for c in target_chains)

    similar_chains = sorted(c.chain for c in similar_structure.chains)
    c2_selection = ",".join(f"{c}" for c in similar_chains)

    Path(similar_structure.probis_compare).mkdir(parents=True, exist_ok=True)

    nosql_path = similar_structure.probis_nosql

    if (nosql_path.exists()):
        return

    cmd = [
        "./probis",
        "-compare",
        "-f1", str(target_structure.probis_srf),
        "-c1", c1_selection,
        "-f2", str(similar_structure.probis_srf),
        "-c2", c2_selection,
        "-out", str(similar_structure.probis_compare),
        "-param", "parameters.imp",
        "-nosql", "probis.nosql",
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    progress.update(
        main_task,
        description=f"{" " * indent}✔ Compared {target_structure.struct_id} with {similar_structure.struct_id}"
    )

    progress.advance(main_task)

    return


def get_results(target_structure: Structure, similar_structure: Structure, progress: Progress | None = None, indent=3):
    main_task = progress.add_task(
        f"{" " * indent}↳ Fetching results for {similar_structure.struct_id}...",
        total=1
    )

    if not target_structure.structure_path.exists():
        raise FileNotFoundError(target_structure.structure_path)
    if not similar_structure.structure_path.exists():
        raise FileNotFoundError(similar_structure.structure_path)

    target_chains = sorted(c.chain for c in target_structure.chains)
    c1_selection = ",".join(f"{c}" for c in target_chains)

    similar_chains = sorted(c.chain for c in similar_structure.chains)
    c2_selection = ",".join(f"{c}" for c in similar_chains)

    if similar_structure.probis_results and similar_structure.probis_results.exists():
        return

    cmd = [
        "./probis",
        "-results",
        "-f1", str(target_structure.structure_path),
        "-c1", c1_selection,
        "-f2", str(similar_structure.structure_path),
        "-c2", c2_selection,
        "-out", str(similar_structure.probis_compare),
        "-nosql", str(similar_structure.probis_nosql),
        "-json", "results.json"
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    progress.update(
        main_task,
        description=f"{" " * indent}✔ Fetched results for {similar_structure.struct_id}"
    )
    progress.advance(main_task)

    return

def get_ligands_for_proteins(uniprot_ids: list[str]) -> List[LigandRecord]:
    placeholders = ",".join(["?"] * len(uniprot_ids))

    query = f"""
    SELECT
        a.protein_id,
        l.inchikey,
        l.ligand_id,
        l.smiles,
        a.affinity_type,
        a.affinity_value
    FROM activity a
    JOIN ligand l ON l.ligand_id = a.ligand_id
    WHERE a.protein_id IN ({placeholders})
      AND a.affinity_value IS NOT NULL
      AND a.affinity_type IN ('Ki','Kd','IC50','EC50')
    ORDER BY a.protein_id, a.p_affinity DESC
    """

    BASE_DIR = Path(__file__).resolve().parent
    DB_PATH = BASE_DIR / "resources" / "ligand.db"

    conn = sqlite3.connect(DB_PATH)

    cur = conn.execute(query, uniprot_ids)
    rows = cur.fetchall()

    return [
        LigandRecord(
            protein_id=r[0],
            inchi_key=r[1],
            ligand_id=r[2],
            smiles=r[3],
            affinity_type=r[4],
            affinity=float(r[5]),
        )
        for r in rows
    ]

def get_fda_approved_ligands() -> List[LigandRecord]:
    query = """
SELECT
    a.protein_id,
    l.inchikey,
    l.ligand_id,
    l.smiles,
    a.affinity_type,
    a.affinity_value
FROM (
    SELECT
        a2.protein_id,
        a2.ligand_id,
        a2.affinity_type,
        a2.affinity_value,
        ROW_NUMBER() OVER (
            PARTITION BY a2.ligand_id
            ORDER BY a2.p_affinity DESC
        ) as rn
    FROM activity a2
    JOIN fda_approved_ligands fda
        ON fda.ligand_id = a2.ligand_id
    WHERE a2.affinity_value IS NOT NULL
      AND a2.affinity_type IN ('Ki','Kd','IC50','EC50')
) a
JOIN ligand l ON l.ligand_id = a.ligand_id
WHERE a.rn = 1;"""

    BASE_DIR = Path(__file__).resolve().parent
    DB_PATH = BASE_DIR / "resources" / "ligand.db"

    conn = sqlite3.connect(DB_PATH)

    cur = conn.execute(query)
    rows = cur.fetchall()

    return [
        LigandRecord(
            protein_id=r[0],
            inchi_key=r[1],
            ligand_id=r[2],
            smiles=r[3],
            affinity_type=r[4],
            affinity=float(r[5]),
        )
        for r in rows
    ]

class ScreeningHit:
    def __init__(self, ligand, pharm2d=None, pharm3d=None, combined_score=None):
        self.ligand = ligand
        self.pharm2d = pharm2d
        self.pharm3d = pharm3d
        self.combined_score = combined_score

def fetch_ligands_for_protein(uniprot_id: str) -> List[Ligand]:
    """
    Fetch training ligands for a single protein from the database.

    For each unique ligand_id, returns exactly one activity record:
    - The record with the highest p_affinity for that (protein, ligand) pair
    - Filters to affinity types: Ki, Kd, IC50, EC50
    - Ensures affinity_value is not NULL
    - Joins with ligand table to get smiles and inchikey

    Args:
        uniprot_id: UniProt ID (protein_id in database)

    Returns:
        List of Ligand objects with LigandRecord data
    """
    query = """
    SELECT DISTINCT
        a.protein_id,
        l.inchikey,
        l.ligand_id,
        l.smiles,
        a.affinity_type,
        a.affinity_value
    FROM (
        SELECT
            a2.protein_id,
            a2.ligand_id,
            a2.affinity_type,
            a2.affinity_value,
            ROW_NUMBER() OVER (
                PARTITION BY a2.protein_id, a2.ligand_id
                ORDER BY a2.p_affinity DESC
            ) as rn
        FROM activity a2
        WHERE a2.protein_id = ?
          AND a2.affinity_value IS NOT NULL
          AND a2.affinity_type IN ('Ki','Kd','IC50','EC50')
    ) a
    JOIN ligand l ON l.ligand_id = a.ligand_id
    WHERE a.rn = 1
    """

    BASE_DIR = Path(__file__).resolve().parent
    DB_PATH = BASE_DIR / "resources" / "ligand.db"

    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(query, (uniprot_id,))
    rows = cur.fetchall()
    conn.close()

    ligands = []
    for r in rows:
        record = LigandRecord(
            protein_id=r[0],
            inchi_key=r[1],
            ligand_id=r[2],
            smiles=r[3],
            affinity_type=r[4],
            affinity=float(r[5]),
        )
        ligands.append(Ligand(record=record))

    return ligands

def screen_ranked_proteins(pipeline: PipelineState, fda_approved_ligands: List[Ligand], progress) -> list[dict]:
    """
    Virtual screening pipeline for ranked proteins using 2D and 3D pharmacophore models.

    For each protein in pipeline.ranked_proteins (sorted by probis_score):
    1. Fetch training ligands from database (one best activity per ligand)
    2. Build Pharm2D consensus from training ligands
    3. Screen all FDA-approved ligands with Pharm2D
    4. Rank by combined score (0.6 Pharm2D + 0.4 Morgan) and keep top 25%
    5. Build Pharm3D consensus from training ligands
    6. Screen filtered FDA ligands with Pharm3D
    7. Compile results into JSON structure per protein

    Args:
        pipeline: PipelineState containing ranked_proteins (sorted by probis_score)
        fda_approved_ligands: List of Ligand objects (FDA-approved compounds)

    Returns:
        List of dicts with structure:
        {
            "protein": str (uniprot_id),
            "probis_score": float,
            "num_training_ligands": int,
            "hits": [
                {
                    "ligand_id": int,
                    "smiles": str,
                    "training_affinity_type": str,
                    "training_affinity_value": float,
                    "pharm2d_similarity": float,
                    "morgan_similarity": float,
                    "combined_2d_score": float,
                    "pharm3d_score": float or null
                }
            ]
        }
    """
    results = []

    if not pipeline.ranked_proteins:
        return results

    screen_task = progress.add_task(
        f"   ↳ Screening {len(pipeline.ranked_proteins)} proteins...",
        total=len(pipeline.ranked_proteins)
    )

    # Initialize screening objects
    pharm2d = Pharm2DScreening(Pharm2DConfig())
    pharm3d = Pharm3DScreening(Pharm3DConfig(feature_consensus_ratio=0.4, proximity=2.5, pair_distance_support_ratio=0.2, pairwise_bounds_k=3.0, num_conformers=20, energy_window=10.0, progress=progress))

    # Precompute FDA ligand fingerprints (one-time)
    pharm2d.set_screen_dataset(fda_approved_ligands)

    for protein in pipeline.ranked_proteins:
        uniprot_id = protein.uniprot_id
        probis_score = protein.probis_score

        progress.update(
            screen_task,
            description=f"   ↳ Screening protein {uniprot_id} (fetching training ligands)..."
        )

        # Fetch training ligands for this protein (highest p_affinity per ligand)
        training_ligands = fetch_ligands_for_protein(uniprot_id)

        if not training_ligands:
            # Skip proteins with no training ligands
            progress.advance(screen_task)
            continue

        num_training = len(training_ligands)

        progress.update(
            screen_task,
            description=f"   ↳ {uniprot_id}: building Pharm2D consensus ({num_training} ligands)"
        )

        # === Stage 1: Pharm2D Screening ===
        # Build consensus from training ligands
        pharm2d.set_consensus(training_ligands)

        progress.update(
            screen_task,
            description=f"   ↳ {uniprot_id}: Pharm2D screening {len(fda_approved_ligands)} FDA ligands"
        )

        # Screen FDA library (pre-loaded dataset)
        pharm2d_results = pharm2d.compare_many_to_many()

        # Rank by combined score and filter to top 25%
        hits = []
        for lig, pharm2d_result in pharm2d_results:
            combined_score = 0.6 * pharm2d_result.pharm2d_similarity + 0.4 * pharm2d_result.morgan_similarity
            hit = ScreeningHit(
                ligand=lig,
                pharm2d=pharm2d_result,
                combined_score=combined_score
            )
            hits.append(hit)

        hits.sort(key=lambda h: h.combined_score, reverse=True)
        num_to_keep = max(1, int(0.25 * len(hits)))
        hits = hits[:num_to_keep]

        # === Stage 2: Pharm3D Screening ===
        # Build consensus from training ligands

        pharm3d_training = select_pharm3d_training_set(training_ligands)

        if len(pharm3d_training) < 2:
            progress.update(
                screen_task,
                description=f"   ↳ {uniprot_id}: skipping Pharm3D (only {len(pharm3d_training)} ligand)"
            )
        else:
            progress.update(
                screen_task,
                description=f"   ↳ {uniprot_id}: Pharm3D consensus ({len(pharm3d_training)} ligands)"
            )

            pharm3d.set_consensus(pharm3d_training)

            pharm3d.set_screen_dataset([h.ligand for h in hits])

            progress.update(
                screen_task,
                description=f"   ↳ {uniprot_id}: Pharm3D screening {len(hits)} candidates"
            )

            pharm3d_results = pharm3d.compare_many_to_many()

            pharm3d_by_ligand_id = {
                lig.record.ligand_id: res
                for lig, res in pharm3d_results
            }

            for hit in hits:
                lig_id = hit.ligand.record.ligand_id
                if lig_id in pharm3d_by_ligand_id:
                    hit.pharm3d = pharm3d_by_ligand_id[lig_id]

        # === Build Output ===
        # Sort hits by Pharm3D score (ascending; lower = better)
        def get_pharm3d_score(hit):
            if hit.pharm3d is None or not hit.pharm3d.embeddings:
                return float('inf')
            return min(e.score for e in hit.pharm3d.embeddings)

        hits_sorted = sorted(hits, key=get_pharm3d_score)

        # Build hit records
        hit_records = []
        for hit in hits_sorted:
            pharm3d_score = None
            if hit.pharm3d is not None and hit.pharm3d.embeddings:
                pharm3d_score = min(e.score for e in hit.pharm3d.embeddings)

            # Get training affinity info for this ligand (from training ligands)
            training_affinity_type = None
            training_affinity_value = None
            for train_lig in training_ligands:
                if train_lig.record.ligand_id == hit.ligand.record.ligand_id:
                    training_affinity_type = train_lig.record.affinity_type
                    training_affinity_value = train_lig.record.affinity
                    break

            hit_record = {
                "ligand_id": hit.ligand.record.ligand_id,
                "smiles": hit.ligand.record.smiles,
                "training_affinity_type": training_affinity_type,
                "training_affinity_value": training_affinity_value,
                "pharm2d_similarity": hit.pharm2d.pharm2d_similarity,
                "morgan_similarity": hit.pharm2d.morgan_similarity,
                "combined_2d_score": hit.combined_score,
                "pharm3d_score": pharm3d_score
            }
            hit_records.append(hit_record)

        # Build protein result
        protein_result = {
            "protein": uniprot_id,
            "probis_score": probis_score,
            "num_training_ligands": num_training,
            "hits": hit_records
        }

        results.append(protein_result)

        progress.advance(screen_task)

    progress.update(
        screen_task,
        description="   ✔ Pharmacophore screening complete"
    )

    return results

def select_pharm3d_training_set(training_ligands: List[Ligand]):
    if not training_ligands or len(training_ligands) < 2:
        return []

    ligs = sorted(
        training_ligands,
        key=lambda l: l.record.affinity or 0,
        reverse=True
    )

    n = len(ligs)

    min_ligands = 10
    percent = int(n * 0.3)
    max_ligands = 30

    k = max(min_ligands, percent)
    k = min(k, max_ligands, n)

    if k < 2:
        return []

    return ligs[:k]

def fetch_sifts_json(pdb_id):
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
    r = session.get(url)
    r.raise_for_status()
    return r.json()


def main(uniprot_id):
    with progress:

        main_task = progress.add_task(
            f"[bold]Finding candidate ligands for {uniprot_id}...",
            total=6
        )

        uniprot_json: UniProtEntry = fetch_uniprot_json_from_id(uniprot_id)

        other_accessions = parse_uniparc_json(get_uniparc_json(uniprot_json.extraAttributes.uniParcId))

        target_protein = ProteinData(
            uniprot_id=uniprot_id,
            uniparc_id=uniprot_json.extraAttributes.uniParcId,
            other_accessions=other_accessions
        )

        prepare_protein_metadata(target_protein, uniprot_json, progress)

        pipeline = PipelineState(target_protein=target_protein)

        pipeline.target_protein.best_structure = select_best_structure(pipeline.target_protein, progress)
        progress.advance(main_task)

        results: List[UniProtEntry] = manage_interpro(pipeline.target_protein.annotations.interpro_ids, progress)
        progress.advance(main_task)

        surface_extraction(pipeline.target_protein.best_structure, progress, indent=3)
        progress.advance(main_task)

        enrichment_task = progress.add_task(
            f"   ↳ Enriching {len(results)} homolog proteins with metadata...",
            total=len(results)
        )

        for entry in results:
            progress.update(enrichment_task,
                            description=f"   ↳ Enriching homolog {entry.primaryAccession} with metadata...")

            other_accessions = parse_uniparc_json(get_uniparc_json(entry.extraAttributes.uniParcId))

            protein = ProteinData(uniprot_id=entry.primaryAccession, uniparc_id=entry.extraAttributes.uniParcId, other_accessions=other_accessions)

            prepare_protein_metadata(protein, entry, None, indent=6)
            pipeline.candidate_proteins.append(protein)

            #TODO: Handle no sifts mapping

            progress.advance(enrichment_task)

        progress.update(enrichment_task, description=f"   ✔ Enriched homolog proteins with metadata")

        ligands_task = progress.add_task(
            f"   ↳ Finding ligands for {len(pipeline.candidate_proteins)} homolog proteins...",
            total=len(pipeline.candidate_proteins)
        )

        uniprot_ids = [p.uniprot_id for p in pipeline.candidate_proteins]

        progress.update(ligands_task, description="   ↳ Querying ligand database (batch)...")

        ligand_records: list[LigandRecord] = get_ligands_for_proteins(uniprot_ids)

        print("[DEBUG] Retrieved ligand records:", len(ligand_records))

        ligands_by_protein: dict[str, list[Ligand]] = defaultdict(list)

        for record in ligand_records:
            ligand_obj = Ligand(record=record)
            ligands_by_protein[record.protein_id].append(ligand_obj)

        filtered_proteins = []

        for entry in pipeline.candidate_proteins:

            accession = entry.uniprot_id

            progress.update(
                ligands_task,
                description=f"   ↳ Mapping ligands to homolog {accession}..."
            )

            ligands_for_protein = ligands_by_protein.get(accession, [])

            if not ligands_for_protein:
                progress.advance(ligands_task)
                continue

            entry.ligands = ligands_for_protein
            entry.ligand_count = len(ligands_for_protein)

            filtered_proteins.append(entry)

            progress.advance(ligands_task)

        pipeline.filtered_proteins = filtered_proteins

        progress.update(
            ligands_task,
            description="   ✔ Found ligands for homolog proteins"
        )

        homologs_task = progress.add_task(
            f"   ↳ Processing {len(pipeline.filtered_proteins)} homolog proteins...",
            total=len(pipeline.filtered_proteins)
        )

        for similar_protein in pipeline.filtered_proteins:
            similar_accession = similar_protein.uniprot_id

            if not similar_protein.structures:
                progress.advance(homologs_task)
                continue

            progress.update(
                homologs_task,
                description=f"   ↳ Processing homolog {similar_accession} (selecting best structure from {len(similar_protein.structures)} candidates)..."
            )

            similar_protein.best_structure = select_best_structure(similar_protein, None, indent=6)

            progress.update(
                homologs_task,
                description=f"   ↳ Processing homolog {similar_accession} (extracting surface of best structure)..."
            )

            surface_extraction(similar_protein.best_structure, None, indent=6)

            progress.update(
                homologs_task,
                description=f"   ↳ Processing homolog {similar_accession} (comparing surfaces)..."
            )

            compare_surfaces(pipeline.target_protein.best_structure, similar_protein.best_structure, indent=6)

            progress.update(
                homologs_task,
                description=f"   ↳ Processing homolog {similar_accession} (obtaining results)..."
            )

            parsed = manage_results(pipeline.target_protein.best_structure, similar_protein.best_structure, None,
                                    indent=6)

            if not parsed:
                progress.advance(homologs_task)
                continue

            similar_protein.probis_score = probis_similarity_score(parsed)

            progress.advance(homologs_task)

        scored = [p for p in pipeline.filtered_proteins if p.probis_score is not None]

        pipeline.ranked_proteins = sorted(
            scored,
            key=lambda item: item.probis_score,
            reverse=True
        )

        print("[INFO] Loading FDA-approved ligands...")
        fda_records = get_fda_approved_ligands()
        fda_approved_ligands = [Ligand(record=r) for r in fda_records]

        print(f"[INFO] Loaded {len(fda_approved_ligands)} FDA-approved ligands")

        print(f"[INFO] Screening {len(pipeline.ranked_proteins)} ranked proteins...")

        final_results = screen_ranked_proteins(pipeline, fda_approved_ligands, progress)

        with open("ranked_protein_virtual_screening.json", "w") as f:
            json.dump(final_results, f, indent=4)

        print("[INFO] Results written to ranked_protein_virtual_screening.json")


#Add optional weighting from CLI via flags
#Add pharmacophore modeling and scaffold hopping (tier 2, lower confidence hits but may be novel)
#Add existing-ligand-similarity for target protein (instead of finding similar proteins, find similar ligands) (tier 3, lowest confidence hits)
#Add two new input methods (custom pdb + residue range, existing pbd id + residue range)
#Use dpocket for holo structures, using explicitp, but use fpocket for apo structures as dpocket requires a ligand
#Check if active site residues/binding site residues from annotations overlap in probis score to boost score
#Handle >1 ligand in complex with protein for holo structures (how to choose relevant pocket etc?)
#Check if author pdb numbering is used in SIFTS or normal pdb numbering

if __name__ == "__main__":
    if len(sys.argv) > 1:
        uniprot = sys.argv[1]
        main(uniprot)
    else:
        print("Usage: python main.py uniprot_id")
