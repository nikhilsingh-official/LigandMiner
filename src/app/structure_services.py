from __future__ import annotations

import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

import gemmi
import numpy as np
import pandas as pd
from Bio.PDB import NeighborSearch, PDBParser
from Bio.PDB.Atom import Atom as BioAtom
from Bio.PDB.Residue import Residue as PDBResidue

from src.domain.ligand import Ligand
from src.domain.pipeline import (
    HasAnnotatedStructures,
    HasIdentifiedStructures,
    HasStructures,
    ProteinStructures,
)
from src.domain.structure import ExperimentalStructure, Pocket, Structure

from .context import session
from .progress import NullProgressSink, ProgressSink
from .protein_services import StructureMetadataService, UniProtService


ARTIFACT_BLACKLIST = {
    "SO4", "PO4", "HPO", "CIT", "ACT", "ACY", "FMT", "NO3", "CL", "BR", "IOD",
    "NA", "K", "MG", "CA", "ZN", "MN", "CU", "NI", "FE", "CO",
    "GOL", "EDO", "PEG", "PGE", "PG4", "PG5", "PG6", "MPD", "DMS", "IPA", "EOH",
    "AMS", "LIT", "TAR", "MAL", "SUC", "TRS", "HEP", "MES", "BIS", "BIC",
    "IMD", "TLA", "ACE", "EPE", "P6G", "P33", "PG0",
    "BOG", "DDM", "OG", "LDA", "SDS", "CHS",
    "DTT", "BME", "TCEP", "HOH",
}

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

BINDING_SITE_WEIGHTS = {
    "domain_coverage": 0.45,
    "active_site_factor": 0.30,
    "geometry": 0.15,
    "physical_sanity": 0.10,
}


class AminoAcid(Enum):
    ALA = "ALA"
    ARG = "ARG"
    ASN = "ASN"
    ASP = "ASP"
    CYS = "CYS"
    GLN = "GLN"
    GLU = "GLU"
    GLY = "GLY"
    HIS = "HIS"
    ILE = "ILE"
    LEU = "LEU"
    LYS = "LYS"
    MET = "MET"
    PHE = "PHE"
    PRO = "PRO"
    SER = "SER"
    THR = "THR"
    TRP = "TRP"
    TYR = "TYR"
    VAL = "VAL"
    MSE = "MSE"
    HSD = "HSD"
    HSE = "HSE"
    HSP = "HSP"

    @classmethod
    def three_letter_from_one_letter(cls, code: str) -> "AminoAcid":
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
        return mapping.get(self, "X")


@dataclass
class StructureResidue(PDBResidue):
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


class StructureService:
    def __init__(
        self,
        session_obj=session,
        uniprot_service: UniProtService | None = None,
        metadata_service: StructureMetadataService | None = None,
    ):
        self.session = session_obj
        self.uniprot_service = uniprot_service or UniProtService(session_obj)
        self.metadata_service = metadata_service or StructureMetadataService(session_obj)

    def clean_structure(self, input_cif, output_pdb, structure):
        input_path = Path(input_cif)
        output_path = Path(output_pdb)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        selection = self.format_chains(structure)

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
            stderr=subprocess.DEVNULL,
        )

    def build_neighbor_search(self, gemmi_structure):
        bio_atoms = []

        for model in gemmi_structure:
            for chain in model:
                for res in chain:
                    for atom in res:
                        coord = atom.pos.tolist()
                        bio_atom = BioAtom(
                            atom.name,
                            np.array(coord),
                            atom.b_iso,
                            atom.occ,
                            " ",
                            atom.name,
                            0,
                            element=atom.element.name.upper(),
                        )
                        bio_atom.chain_id = chain.name
                        bio_atom.residue = res
                        bio_atoms.append(bio_atom)

        return NeighborSearch(bio_atoms), bio_atoms

    def ligand_passes_basic_filters(self, residue, structure):
        comp_id = residue.name

        if comp_id in ARTIFACT_BLACKLIST:
            return False

        copies = sum(r.name == comp_id for m in structure for c in m for r in c)
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

    def get_pocket_residues(self, ligand_residue, ns):
        contact_distance = 5.0
        pocket = set()

        for atom in ligand_residue:
            neighbors = ns.search(atom.pos.tolist(), contact_distance)

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

    def search_pubchem_by_smiles(self, smiles: str) -> dict | None:
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/JSON"
        response = self.session.get(url, params={"smiles": smiles})
        if response.status_code == 200:
            props = response.json()["PC_Compounds"][0]["props"]
            names = [p for p in props if p["urn"]["label"] == "IUPAC Name"]
            return names[0]["value"]["sval"] if names else None
        return None

    def is_holo(self, input_cif, valid_chains):
        structure = gemmi.read_structure(str(input_cif))
        ns, _ = self.build_neighbor_search(structure)
        valid_chain_ids = {c.chain if hasattr(c, "chain") else c for c in valid_chains}
        pocket_candidates = []

        for model in structure:
            for chain in model:
                for res in chain:
                    if res.het_flag != "H":
                        continue

                    if not self.ligand_passes_basic_filters(res, structure):
                        continue

                    pocket = self.get_pocket_residues(res, ns)

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

    def download_structure(self, structure: Structure, progress: ProgressSink, task_key: str, root):
        struct_id = structure.struct_id.upper()
        progress.update(task_key, f"Downloading structures ({struct_id})...")

        structure.paths.raw_structure_path.parent.mkdir(parents=True, exist_ok=True)

        if structure.paths.raw_structure_path.exists():
            progress.advance(task_key)
            return structure

        res = self.session.get(structure.cif)

        if res.status_code != 200:
            raise RuntimeError(f"Failed to download {struct_id} (HTTP {res.status_code})")

        with open(structure.paths.raw_structure_path, "wb") as f:
            f.write(res.content)

        progress.advance(task_key)
        return structure

    def download_structures(
        self,
        protein: HasStructures,
        progress: ProgressSink,
        task_key: str,
        root="./",
        max_workers=6,
    ):
        downloaded = []
        root = Path(root)
        structures = protein.structures.structures

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    self.download_structure,
                    structure,
                    progress,
                    task_key,
                    root,
                ): structure
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

    def residues_to_fpocket_p(self, residues):
        parts = []

        for chain, resseq, icode in residues:
            icode = icode if icode else "-"
            parts.append(f"{resseq}:{icode}:{chain}")

        return ".".join(parts)

    def run_fpocket(self, structure: Structure) -> Path:
        pdb_path = structure.paths.structure_path
        desired_outdir = structure.paths.fpocket_out
        work_dir = pdb_path.parent

        true_pocket = structure.analysis.true_pocket
        fpocket_p = self.residues_to_fpocket_p(true_pocket)

        subprocess.run(
            ["fpocket", "-f", pdb_path.name, "--custom-pocket", fpocket_p],
            cwd=work_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        generated_outdir = work_dir / f"{pdb_path.stem}_out"

        if not generated_outdir.exists():
            raise RuntimeError("fpocket output directory was not created.")

        if desired_outdir.exists():
            shutil.rmtree(desired_outdir)

        shutil.move(str(generated_outdir), str(desired_outdir))
        return desired_outdir

    def parse_fpocket_info(self, fpocket_out: Path) -> list[Pocket]:
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

    def format_chains(self, structure):
        chain_ids = sorted(c.chain for c in structure.chains)
        return ",".join(f"{c}" for c in chain_ids)

    def pdb_residue_exists(self, chain, resseq):
        return (" ", resseq, " ") in chain

    def has_sidechain(self, res):
        return any(atom.get_id() not in ("N", "CA", "C", "O") for atom in res.get_atoms())

    def normalize_resolution(self, res):
        res = min(max(res, 1.0), 3.5)
        return 1.0 - (res - 1.0) / (3.5 - 1.0)

    def has_valid_binding_pocket(self, pocket_scores):
        return any(score > 0.0 for score in pocket_scores)

    def extract_pocket_residues(self, pocket_cif):
        st = gemmi.read_structure(pocket_cif)
        residues = set()

        for model in st:
            for chain in model:
                for res in chain:
                    residues.add((chain.name, res.seqid.num))

        return sorted(residues)

    def get_chains_from_motif(self, motif: str) -> str:
        chains = sorted(set(re.findall(r":([A-Za-z0-9])", motif)))

        if not chains:
            raise ValueError(f"No chains found in motif: {motif}")

        return ",".join(chains)

    def expand_residues_spatially(self, pdb_path: str, seed_residues, radius: float = 8.0):
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

    def residues_to_motif(self, structure):
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

    def fpocket_enrichment(self, data: HasStructures):
        structures = data.structures.structures
        for structure in structures:
            struct_id = structure.struct_id.upper()

            if not structure.paths.structure_path.exists():
                print(f"[FPOCKET SKIP] Missing PDB for {struct_id}")
                structure.analysis.fpockets = []
                continue

            if structure.analysis.fpockets:
                continue

            if structure.paths.fpocket_txt.exists() and structure.paths.fpocket_txt.stat().st_size > 0:
                try:
                    structure.analysis.fpockets = self.parse_fpocket_info(structure.paths.fpocket_out)
                    continue
                except Exception as e:
                    print(f"[FPOCKET CACHE PARSE FAIL] {struct_id}: {e}")
                    structure.analysis.fpockets = []

            if structure.paths.fpocket_out.exists():
                shutil.rmtree(structure.paths.fpocket_out)

            try:
                self.run_fpocket(structure)
                structure.analysis.fpockets = self.parse_fpocket_info(structure.paths.fpocket_out)
            except Exception as e:
                print(f"[FPOCKET RUN FAIL] {struct_id}: {e}")
                structure.analysis.fpockets = []

        return data

    def parse_fpocket_atm_pdb(self, pdb_path):
        st = gemmi.read_structure(str(pdb_path))
        residues = set()

        for model in st:
            for chain in model:
                for residue in chain:
                    if residue.name == "":
                        continue

                    resseq = residue.seqid.num
                    icode = residue.seqid.icode
                    chain_id = chain.name
                    residues.add((chain_id, resseq, icode))

        return sorted(residues)

    def geometry_score(self, pocket):
        volume = min(pocket.get("Volume", 0.0) / 400.0, 1.0)
        alpha_density = min(pocket.get("Alpha sphere density", 0.0) / 3.0, 1.0)
        volume_score = min(pocket.get("Volume score", 0.0) / 4.0, 1.0)
        return (volume + alpha_density + volume_score) / 3.0

    def physical_sanity_score(self, pocket):
        rigidity = max(1.0 - pocket.get("Flexibility", 1.0), 0.0)

        solvent_access = pocket.get("Mean alp. sph. solvent access", 1.0)
        burial = max(1.0 - solvent_access, 0.0)

        polar_frac = pocket.get("Proportion of polar atoms", 0.0) / 100.0
        polarity_ok = max(1.0 - abs(polar_frac - 0.5), 0.0)

        return 0.4 * rigidity + 0.3 * burial + 0.3 * polarity_ok

    def compute_binding_site_score(self, pocket):
        active_site = pocket.get("active_site_coverage", 0.0)
        purity = pocket.get("pocket_domain_purity", 0.0)

        geo = self.geometry_score(pocket)
        phys = self.physical_sanity_score(pocket)

        score = geo * phys * (1.0 + 2.5 * active_site) * (0.5 + purity)
        return round(score, 4)

    def score_pockets(
        self,
        protein: HasAnnotatedStructures,
        *,
        residue_sasa=None,
        min_surface_residues=6,
        min_surface_sasa=5.0,
    ):
        domains = protein.annotations.annotations.domains
        active_sites = protein.annotations.annotations.active_sites
        structures = protein.structures.structures

        main_domain = max(domains, key=lambda d: d.end - d.start, default=None)
        domain_residues = (
            set(range(main_domain.start, main_domain.end + 1))
            if main_domain
            else set()
        )

        active_site_residues = {
            r for site in active_sites for r in range(site.start, site.end + 1)
        }

        for structure in structures:
            for pocket in structure.analysis.fpockets:
                residues = pocket.uniprot_indices

                if not residues or not domain_residues:
                    continue

                pocket_res = {r[1] for r in residues}

                if residue_sasa is not None:
                    surface_res = {
                        r for r in pocket_res if residue_sasa.get(r, 0.0) >= min_surface_sasa
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
                    if active_site_residues
                    else 0.0
                )

                druggability = pocket.druggability_score

                pocket.binding_site_score = (
                    0.30 * pocket.pocket_domain_purity
                    + 0.25 * pocket.active_site_coverage
                    + 0.25 * druggability
                    + 0.20 * surface_fraction
                )

        return protein

    def pocket_residue_enrichment(self, protein: HasIdentifiedStructures):
        for structure in protein.structures.structures:
            for pocket in structure.analysis.fpockets:
                path = structure.paths.fpocket_pockets / f"pocket{pocket.pocket_id}_atm.pdb"
                pocket.residues = self.parse_fpocket_atm_pdb(path)

        return self.map_residues_to_uniprot(protein)

    def map_residues_to_uniprot(self, protein: HasIdentifiedStructures):
        for structure in protein.structures.structures:
            use_sifts = isinstance(structure, ExperimentalStructure)
            sifts = structure.analysis.sifts_mapping if use_sifts else None

            for pocket in structure.analysis.fpockets:
                mapped_residues = []

                for chain, resseq, icode in pocket.residues:
                    if use_sifts and sifts and chain == sifts["chain_id"]:
                        unp_res = self.uniprot_service.map_pdb_to_uniprot(resseq, sifts)

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

    def get_pocket_from_prank(self, structure: Structure):
        subprocess.run(
            [
                "prank",
                "predict",
                "-f", str(structure.paths.structure_path),
                "-o", str(structure.paths.prank_out),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        df = pd.read_csv(structure.paths.prank_csv, skipinitialspace=True)

        if df.empty:
            return None, None

        best_row = df.loc[df["probability"].idxmax()]
        probability = float(best_row["probability"])
        residue_ids_str = best_row["residue_ids"]
        residues = []

        for token in residue_ids_str.split():
            chain, resseq_str = token.split("_")
            resseq = int("".join(c for c in resseq_str if c.isdigit()))
            icode = "".join(c for c in resseq_str if c.isalpha())
            residues.append((chain, resseq, icode))

        return probability, residues

    def shortlist_structures(self, structures: list[Structure], top_n=100) -> list[Structure]:
        structures.sort(key=lambda s: s.analysis.structural_plausibility or 0.0, reverse=True)
        return structures[:top_n]

    def select_best_structure(
        self,
        protein: HasAnnotatedStructures,
        progress: ProgressSink | None = None,
    ):
        progress = progress or NullProgressSink()

        progress.start(
            "select_best_structure",
            f"Selecting best structure for {protein.uniprot_id}...",
            total=5,
        )

        enriched_structures = self.metadata_service.metadata_structures_enrichment(
            protein.structures.structures,
            protein.annotations.annotations.domains,
            protein.annotations.annotations.active_sites,
            protein.uniprot_id,
        )
        protein.structures = ProteinStructures(
            structures=enriched_structures,
            best_structure=protein.structures.best_structure,
        )

        protein.structures.structures = self.shortlist_structures(
            protein.structures.structures,
            top_n=100,
        )

        progress.update(
            "select_best_structure",
            f"Selecting best structure for {protein.uniprot_id} (downloading structures)…",
        )

        progress.start(
            "download_structures",
            "Downloading structures...",
            total=len(protein.structures.structures),
        )

        self.download_structures(protein, progress, "download_structures", "structures")

        progress.finish("download_structures", "[green]Downloaded structures")
        progress.advance("select_best_structure")

        progress.update(
            "select_best_structure",
            f"Selecting best structure for {protein.uniprot_id} (preparing structures)…",
        )
        progress.start(
            "prepare_structures",
            "Preparing structures...",
            total=len(protein.structures.structures),
        )
        for structure in protein.structures.structures:
            progress.update("prepare_structures", f"Preparing structures ({structure.struct_id})...")
            self.clean_structure(structure.paths.raw_structure_path, structure.paths.structure_path, structure)
            structure.analysis.true_pocket = self.is_holo(structure.paths.raw_structure_path, structure.chains)
            if structure.analysis.true_pocket:
                structure.analysis.true_pocket_certainty = 1.0
            else:
                probability, residues = self.get_pocket_from_prank(structure)
                structure.analysis.true_pocket = residues
                structure.analysis.true_pocket_certainty = probability
            progress.advance("prepare_structures")

        progress.finish("prepare_structures", "[green]Prepared structures")
        progress.advance("select_best_structure")

        progress.update(
            "select_best_structure",
            f"Selecting best structure for {protein.uniprot_id} (fpocket enrichment)…",
        )
        progress.start("fpocket_enrichment", "Running fpocket...", total=1)
        protein = self.fpocket_enrichment(protein)
        progress.advance("fpocket_enrichment")
        progress.finish("fpocket_enrichment", "[green]Enriched structures by fpocket")
        progress.advance("select_best_structure")

        progress.update(
            "select_best_structure",
            f"Selecting best structure for {protein.uniprot_id} (residue enrichment)…",
        )
        progress.start("residue_enrichment", "Enriching pocket residues...", total=1)
        protein = self.pocket_residue_enrichment(protein)
        progress.advance("residue_enrichment")
        progress.finish("residue_enrichment", "[green]Enriched pocket residues")
        progress.advance("select_best_structure")

        progress.update(
            "select_best_structure",
            f"Selecting best structure for {protein.uniprot_id} (scoring)…",
        )
        progress.start("score_pockets", "Scoring pockets...", total=1)
        protein = self.score_pockets(protein)
        best_structure = max(
            protein.structures.structures,
            key=lambda s: self.metadata_service.structure_selection_score(
                s,
                w_structure=0.7,
                w_pocket=0.3,
            ),
        )

        progress.advance("score_pockets")
        progress.finish("score_pockets", "[green]Structures scored")
        progress.advance("select_best_structure")

        progress.finish(
            "select_best_structure",
            f"Best structure selected for {protein.uniprot_id}: {best_structure.struct_id}",
        )

        return best_structure
