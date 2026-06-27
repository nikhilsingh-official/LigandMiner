from __future__ import annotations

import re
import numpy as np
from Bio.PDB import NeighborSearch
from Bio.PDB.Atom import Atom as BioAtom

from src.domain.ligand import Ligand
from src.domain.pipeline import HasAnnotatedStructures
from src.domain.structure import ExperimentalStructure, Structure

from ..context import session
from .pocket_service import PocketService
from ..progress import NullProgressSink, ProgressSink
from ..protein_services import StructureMetadataService, UniProtService
from .structure_downloader import StructureDownloader


class StructureSelectionService:
    def __init__(
        self,
        session_obj=session,
        uniprot_service: UniProtService | None = None,
        metadata_service: StructureMetadataService | None = None,
        pocket_service: PocketService | None = None,
        structure_downloader: StructureDownloader | None = None,
    ):
        self.session = session_obj
        self.uniprot_service = uniprot_service or UniProtService(session_obj)
        self.metadata_service = metadata_service or StructureMetadataService(session_obj)
        self.pocket_service = pocket_service or PocketService(self.uniprot_service)
        self.structure_downloader = structure_downloader or StructureDownloader(session_obj)
        self.ARTIFACT_BLACKLIST = {
            "SO4", "PO4", "HPO", "CIT", "ACT", "ACY", "FMT", "NO3", "CL", "BR", "IOD",
            "NA", "K", "MG", "CA", "ZN", "MN", "CU", "NI", "FE", "CO",
            "GOL", "EDO", "PEG", "PGE", "PG4", "PG5", "PG6", "MPD", "DMS", "IPA", "EOH",
            "AMS", "LIT", "TAR", "MAL", "SUC", "TRS", "HEP", "MES", "BIS", "BIC",
            "IMD", "TLA", "ACE", "EPE", "P6G", "P33", "PG0",
            "BOG", "DDM", "OG", "LDA", "SDS", "CHS",
            "DTT", "BME", "TCEP", "HOH",
        }

    @staticmethod
    def pdb_residue_exists(chain, resseq):
        return (" ", resseq, " ") in chain

    @staticmethod
    def has_sidechain(res):
        return any(atom.get_id() not in ("N", "CA", "C", "O") for atom in res.get_atoms())

    @staticmethod
    def normalize_resolution(res):
        res = min(max(res, 1.0), 3.5)
        return 1.0 - (res - 1.0) / (3.5 - 1.0)

    @staticmethod
    def get_chains_from_motif(motif: str) -> str:
        chains = sorted(set(re.findall(r":([A-Za-z0-9])", motif)))

        if not chains:
            raise ValueError(f"No chains found in motif: {motif}")

        return ",".join(chains)

    @staticmethod
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

        if comp_id in self.ARTIFACT_BLACKLIST:
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

    def shortlist_structures(self, structures: list[Structure], top_n=100) -> list[Structure]:
        structures.sort(key=lambda s: s.analysis.structural_plausibility or 0.0, reverse=True)
        return structures[:top_n]

    def select_best_structure(
        self,
        protein: HasAnnotatedStructures,
        progress: ProgressSink | None = None,
    ) -> Structure:
        progress = progress or NullProgressSink()

        with progress.group(
            "select_best_structure",
            f"Selecting best structure for {protein.uniprot_id}...",
            total=5,
        ) as group:
            enriched_structures = self.metadata_service.metadata_structures_enrichment(
                protein.structures.structures,
                protein.annotations.annotations.domains,
                protein.annotations.annotations.active_sites,
                protein.uniprot_id,
            )
            protein.structures.structures = enriched_structures

            protein.structures.structures = self.shortlist_structures(
                protein.structures.structures, top_n=100
            )
            group.advance()

            with group.group(
                "download_structures",
                "Downloading structures...",
                total=len(protein.structures.structures),
            ) as download_group:
                self.structure_downloader.download_structures(
                    protein,
                    download_group,
                    "download_structures",
                    "structures",
                )

            group.advance()

            with group.group(
                "prepare_structures",
                "Preparing structures...",
                total=len(protein.structures.structures),
            ) as prepare_group:
                for structure in protein.structures.structures:
                    prepare_group.update(f"Preparing structures ({structure.struct_id})...")
                    self.structure_downloader.clean_structure(
                        structure.paths.raw_structure_path,
                        structure.paths.structure_path,
                        structure,
                    )
                    structure.analysis.true_pocket = self.is_holo(
                        structure.paths.raw_structure_path, structure.chains
                    )
                    if structure.analysis.true_pocket:
                        structure.analysis.true_pocket_certainty = 1.0
                    else:
                        probability, residues = self.pocket_service.get_pocket_from_prank(
                            structure
                        )
                        structure.analysis.true_pocket = residues
                        structure.analysis.true_pocket_certainty = probability

            group.advance()

            with group.group(
                "fpocket_enrichment", "Running fpocket...", total=1
            ) as fpocket_group:
                protein = self.pocket_service.fpocket_enrichment(protein)

            group.advance()

            with group.group(
                "residue_enrichment", "Enriching pocket residues...", total=1
            ) as residue_group:
                protein = self.pocket_service.pocket_residue_enrichment(protein)

            with group.group("score_pockets", "Scoring pockets...", total=1) as score_group:
                protein = self.pocket_service.score_pockets(protein)
                best_structure = max(
                    protein.structures.structures,
                    key=lambda s: self.metadata_service.structure_selection_score(
                        s, w_structure=0.7, w_pocket=0.3
                    ),
                )

            group.advance()

            group.finish(
                f"Best structure selected for {protein.uniprot_id}: {best_structure.struct_id}"
            )

        return best_structure
