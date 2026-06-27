from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import gemmi
import pandas as pd

from src.domain.pipeline import HasAnnotatedStructures, HasIdentifiedStructures, HasStructures
from src.domain.structure import Pocket, Structure

from ..protein_services import UniProtService


class PocketService:
    def __init__(self, uniprot_service: UniProtService):
        self.uniprot_service = uniprot_service
        self.KEY_MAP = {
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

                    if key in self.KEY_MAP:
                        setattr(current, self.KEY_MAP[key], value)

            if current:
                pockets.append(current)

        return pockets

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
