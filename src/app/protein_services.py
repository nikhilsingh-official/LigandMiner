from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import List

from pydantic import TypeAdapter

from src.domain.common import (
    MethodType,
    NON_RESOLUTION_METHODS,
    RESOLUTION_METHODS,
)
from src.domain.pipeline import ProteinAnnotations, ProteinData, ProteinStructures
from src.domain.scoring import (
    ALPHA_FOLD_WEIGHTS,
    PDB_METHOD_SCORE,
    PDB_WEIGHTS,
    STRUCTURE_DB_MULTIPLIER,
    SWISS_MODEL_WEIGHTS,
)
from src.domain.structure import (
    AlphaFoldStructure,
    AlphaFoldStructureMetadata,
    Chain,
    ExperimentalStructure,
    ExperimentalStructureMetadata,
    HomologyStructure,
    HomologyStructureMetadata,
    PropertyData,
    StructurePaths,
    Structure,
)
from src.domain.uniprot import Region, SequenceAnnotations, UniProtEntry, UniProtKBCrossReference, Property

from .context import session
from .progress import NullProgressSink, ProgressSink

STRUCTURES_DIR = Path("structures")
COMPARISON_DIR = Path("probis_compare")
STRUCTURE_TYPE_MULTIPLIER = {
    "PDB": 1.0,
    "SMR": 0.8,
    "AlphaFoldDB": 0.6,
}

PDB_RAW_WEIGHTS = {
    "resolution": 0.70,
    "method": 0.30,
}

VALID_STRUCTURE_DATABASES = {"PDB", "AlphaFoldDB", "SMR"}

class UniProtService:
    def __init__(self, session_obj=session):
        self.session = session_obj

    def fetch_interpro_json(self, interpro_id):
        url = f"https://www.ebi.ac.uk/interpro/api/entry/interpro/{interpro_id}/"
        res = self.session.get(url)
        if res.status_code != 200:
            print(f"Invalid response: {interpro_id}")
            return None
        return res.json()

    def fetch_uniprot_json_from_id(self, uniprot_id: str) -> UniProtEntry:
        url = (
            "https://rest.uniprot.org/uniprotkb/"
            f"{uniprot_id}.json?fields=accession,ft_domain,ft_act_site,ft_binding,"
            "uniparc_id,structure_3d,xref_alphafolddb,xref_smr,xref_interpro"
        )
        res = self.session.get(url)
        res.raise_for_status()
        return UniProtEntry.model_validate(res.json())

    def fetch_uniprot_json_from_interpro(
        self,
        interpro_id: str,
        progress: ProgressSink,
        reviewed: bool = True,
        page_size: int = 500,
    ) -> List[UniProtEntry]:
        progress.start("fetch_uniprot", "Fetching InterPro members' UniProt...", total=None)

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
        completed = 0

        while True:
            progress.update("fetch_uniprot", f"Fetching InterPro members' UniProt (page {page})...")

            res = self.session.get(url, params=params)
            if res.status_code != 200:
                raise RuntimeError(f"UniProt request failed: {res.status_code}")

            data = res.json()
            results.extend(data.get("results", []))

            completed += 1
            progress.advance("fetch_uniprot")

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

        progress.update(
            "fetch_uniprot",
            f"[green]Retrieved {len(results)} UniProt entries",
            total=completed,
            completed=completed,
        )

        adapter = TypeAdapter(List[UniProtEntry])
        return adapter.validate_python(results)

    def parse_chains(self, chain_str):
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
                results.append({"chain": chain, "start": start, "end": end})

        return results

    def map_pdb_to_uniprot(self, resseq: int, mapping: dict) -> int | None:
        pdb_start = mapping["start"]["residue_number"]
        pdb_end = mapping["end"]["residue_number"]

        if not (pdb_start <= resseq <= pdb_end):
            return None

        return mapping["unp_start"] + (resseq - pdb_start)

    def map_uniprot_chains_to_pdb(self, chains):
        pass

    def get_experimental_structure_property_data_from_properties(
        self,
        properties: List[Property]
    ) -> PropertyData:

        metadata = ExperimentalStructureMetadata()
        chains = []

        for prop in properties or []:
            if prop.key == "Method":
                metadata.method = prop.value
            elif prop.key == "Resolution":
                try:
                    metadata.resolution = float(prop.value.replace(" A", ""))
                except Exception:
                    metadata.resolution = None
            elif prop.key == "Chains":
                chains = [
                    Chain(**c) for c in self.parse_chains(prop.value)
                ]

        return PropertyData(metadata=metadata, chains=chains)

    def uniprot_cross_references_to_structures(
        self,
        cross_references: List[UniProtKBCrossReference]
    ):
        refs_by_database = defaultdict(list)

        for ref in cross_references:
            refs_by_database[ref.database].append(ref)

        structures = []

        for db, refs in refs_by_database.items():
            if db == "PDB":
                properties = self.get_experimental_structure_property_data_from_properties(ref.properties)
                structure = ExperimentalStructure(
                    struct_id=ref.id,
                    database=ref.database,
                    metadata=properties.metadata,
                    chains=properties.chains,
                    cif=f"https://files.rcsb.org/download/{ref.id}.cif",
                )
            elif db == "SMR":
                structure = HomologyStructure(
                    struct_id=ref.id,
                    database=ref.database,
                )
            elif db == "AlphaFoldDB":
                structure = AlphaFoldStructure(
                    struct_id=ref.id,
                    database=ref.database
                )
            structures.append(structure)

        return structures

    def parse_uniprot_cross_references(
        self,
        uniprot_json: UniProtEntry,
    ) -> tuple[SequenceAnnotations, List[Structure]]:
        cross_refs = uniprot_json.uniProtKBCrossReferences or []

        structures: List[Structure] = []

        interpro_ids = [
            ref.id for ref in cross_refs if ref.database == "InterPro"
        ]
        annotations = SequenceAnnotations(interpro_ids=interpro_ids)

        for ref in cross_refs:
            if ref.database not in VALID_STRUCTURE_DATABASES:
                continue

            if ref.database == "PDB":
                property_data = self.get_experimental_structure_property_data_from_properties(
                    ref.properties
                )
                structure = ExperimentalStructure(
                    struct_id=ref.id,
                    database=ref.database,
                    cif=f"https://files.rcsb.org/download/{ref.id}.cif",
                    metadata=property_data.metadata,
                    chains=property_data.chains,
                )

            elif ref.database == "SMR":
                structure = HomologyStructure(
                    struct_id=ref.id,
                    database=ref.database,
                )
            else:
                structure = AlphaFoldStructure(
                    struct_id=ref.id,
                    database=ref.database,
                )

            structures.append(structure)

        return annotations, structures

    def parse_uniprot_features(
        self,
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
        self,
        uniprot_json: UniProtEntry,
    ) -> tuple[SequenceAnnotations, List[Structure]]:
        annotations, structures = self.parse_uniprot_cross_references(uniprot_json)
        annotations = self.parse_uniprot_features(uniprot_json, annotations)
        return annotations, structures

    def get_uniparc_json(self, uniparc_id: str):
        url = f"https://www.uniprot.org/uniparc/{uniparc_id}.json"
        res = self.session.get(url, timeout=30)

        if not res.ok:
            raise RuntimeError(
                f"Failed to fetch UniParc data for {uniparc_id} (HTTP {res.status_code})"
            )

        return res.json()

    def parse_uniparc_json(self, json_data) -> List[str]:
        return [
            entry["id"]
            for entry in json_data.get("uniParcCrossReferences", [])
            if entry.get("database").startswith("UniProtKB")
        ]

    def fetch_sifts_json(self, pdb_id):
        url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
        r = self.session.get(url)
        r.raise_for_status()
        return r.json()

    def parse_sifts_json(
        self,
        mapping_json,
        structure: ExperimentalStructure,
        accessions: List[str],
    ):
        pdb_id = structure.struct_id.lower()

        if pdb_id not in mapping_json:
            raise ValueError(f"No SIFTS entry for {structure.struct_id}")

        entry = mapping_json[pdb_id].get("UniProt", {})
        struct_chains = {c.chain for c in structure.chains}

        for accession in accessions:
            if accession not in entry:
                continue

            mappings = entry[accession].get("mappings", [])

            for mapping in mappings:
                label_chain = mapping.get("struct_asym_id")
                auth_chain = mapping.get("chain_id")

                if label_chain in struct_chains or auth_chain in struct_chains:
                    return mapping

        return None

    def prepare_protein_metadata(
        self,
        protein: ProteinData,
        entry: UniProtEntry,
        progress: ProgressSink | None = None,
    ) -> None:
        progress = progress or NullProgressSink()

        progress.start("prepare_metadata", f"Parsing UniProt for {protein.uniprot_id}...", total=1)
        annotations, structures = self.parse_uniprot_json(entry)
        protein.annotations = ProteinAnnotations(annotations=annotations)

        accessions = list(
            dict.fromkeys([protein.uniprot_id] + (protein.other_accessions or []))
        )

        for structure in structures:
            if not isinstance(structure, ExperimentalStructure):
                continue

            mapping = self.fetch_sifts_json(structure.struct_id)

            try:
                structure.analysis.sifts_mapping = self.parse_sifts_json(
                    mapping,
                    structure,
                    accessions,
                )
            except ValueError:
                structure.analysis.sifts_mapping = None

        protein.structures = ProteinStructures(structures=structures)

        progress.advance("prepare_metadata")
        progress.finish("prepare_metadata", f"[green]UniProt parsed for {protein.uniprot_id}")

    def fetch_least_abstracted_interpro_id(self, interpro_ids, progress: ProgressSink):
        progress.start("interpro_group", "Searching for InterPro group...", total=len(interpro_ids))

        least_abstracted = None
        completed = 0

        for interpro_id in interpro_ids:
            interpro_json = self.fetch_interpro_json(interpro_id)

            if interpro_json:
                interpro_type = interpro_json["metadata"]["type"]

                if interpro_type == "family":
                    remaining = len(interpro_ids) - completed
                    progress.update("interpro_group", f"Found InterPro family: {interpro_id}")
                    progress.advance("interpro_group", remaining)
                    return {"id": interpro_id, "type": "family"}

                if interpro_type == "domain" and least_abstracted is None:
                    least_abstracted = {"id": interpro_id, "type": "domain"}

            completed += 1
            progress.advance("interpro_group")

        progress.finish(
            "interpro_group",
            (
                f"Found InterPro {least_abstracted['type']}: {least_abstracted['id']}"
                if least_abstracted
                else "No InterPro family or domain found"
            ),
        )
        return least_abstracted

    def manage_interpro(
        self,
        interpro_ids: list[str],
        uniprot_id: str,
        progress: ProgressSink | None = None,
    ):
        progress = progress or NullProgressSink()

        progress.start(
            "manage_interpro",
            f"Searching for candidate similar proteins to {uniprot_id}...",
            total=2,
        )

        progress.update(
            "manage_interpro",
            f"Searching for candidate similar proteins to {uniprot_id} (fetching InterPro family)…",
        )

        least_abstracted = self.fetch_least_abstracted_interpro_id(
            interpro_ids,
            progress.child("interpro"),
        )

        if least_abstracted is None:
            progress.finish(
                "manage_interpro",
                f"Found 0 candidate similar proteins to {uniprot_id}",
            )
            return []

        progress.update(
            "manage_interpro",
            f"Searching for candidate similar proteins to {uniprot_id} (fetching proteins' UniProt entries)…",
        )

        progress.advance("manage_interpro")

        results: List[UniProtEntry] = self.fetch_uniprot_json_from_interpro(
            least_abstracted["id"],
            progress.child("members"),
        )

        progress.advance("manage_interpro")
        progress.finish(
            "manage_interpro",
            f"Found {len(results)} candidate similar proteins to {uniprot_id}",
        )
        return results


class StructureMetadataService:
    def __init__(self, session_obj=session):
        self.session = session_obj

    def get_swiss_model_intermediate(self, swiss_model_id: str):
        swiss_model_id = swiss_model_id.upper()
        intermediate_url = (
            f"https://swissmodel.expasy.org/repository/uniprot/{swiss_model_id}.json"
        )
        res = self.session.get(intermediate_url)

        if res.status_code != 200:
            raise RuntimeError(
                f"Failed to download {swiss_model_id} (HTTP {res.status_code})"
            )

        return res.json()

    def get_alpha_fold_intermediate(self, alphafold_id: str):
        alphafold_id = alphafold_id.upper()
        intermediate_url = (
            f"https://alphafold.ebi.ac.uk/api/uniprot/summary/{alphafold_id}.json"
        )
        res = self.session.get(intermediate_url)

        if res.status_code != 200:
            raise RuntimeError(
                f"Failed to download {alphafold_id} (HTTP {res.status_code})"
            )

        return res.json()

    def parse_swiss_model_intermediate_json(self, json_data, uniprot: str):
        parsed_structures: list[HomologyStructure] = []

        for idx, structure in enumerate(json_data.get("result", {}).get("structures", [])):
            if structure.get("method") != "HOMOLOGY MODELLING":
                continue

            chains = [
                Chain(
                    chain=chain["id"],
                    start=structure["from"],
                    end=structure["to"],
                )
                for chain in structure.get("chains", [])
            ]

            parsed_structures.append(
                HomologyStructure(
                    struct_id=f"SM-{uniprot}-{idx}",
                    database="SMR",
                    cif=structure.get("modelcif"),
                    chains=chains,
                    metadata=HomologyStructureMetadata(
                        coverage=structure.get("coverage"),
                        gmqe=structure.get("gmqe"),
                        identity=structure.get("identity"),
                        similarity=structure.get("similarity"),
                        avg_local_score=structure.get("qmean", {}).get("avg_local_score"),
                        qmean6_z=structure.get("qmean", {}).get("qmean6_z_score"),
                        provider=structure.get("provider"),
                        template=structure.get("template"),
                        created_date=structure.get("created_date"),
                    ),
                )
            )

        return parsed_structures

    def parse_alpha_fold_intermediate_json(self, json_data, uniprot):
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
                            end=summary.get("uniprot_end"),
                        )
                    )

            parsed_structures.append(
                AlphaFoldStructure(
                    struct_id=f"AF-{uniprot}-{idx}",
                    database="AlphaFoldDB",
                    cif=summary.get("model_url"),
                    chains=chains,
                    metadata=AlphaFoldStructureMetadata(
                        coverage=summary.get("coverage"),
                        confidence_type=summary.get("confidence_type"),
                        avg_local_score=summary.get("confidence_avg_local_score"),
                        provider=summary.get("provider"),
                        created_date=summary.get("created"),
                    ),
                )
            )

        return parsed_structures

    def score_alpha_fold_structure(self, structure: AlphaFoldStructure):
        raw = 0.0

        if structure.metadata.avg_local_score is not None:
            norm_plddt = structure.metadata.avg_local_score / 100.0
            raw += ALPHA_FOLD_WEIGHTS["avg_local_score"] * norm_plddt
        if structure.metadata.coverage is not None:
            raw += ALPHA_FOLD_WEIGHTS["coverage"] * structure.metadata.coverage

        calibrated = raw ** 1.25
        calibrated *= STRUCTURE_DB_MULTIPLIER["AlphaFoldDB"]
        structure.analysis.structural_plausibility = round(calibrated, 4)
        return structure

    def score_alpha_fold_structures(self, structures: list[AlphaFoldStructure]):
        return [self.score_alpha_fold_structure(structure) for structure in structures]

    def score_pdb_structure(self, structure: ExperimentalStructure):
        resolution: float | None = structure.metadata.resolution
        method: MethodType | None = structure.metadata.method

        if method in RESOLUTION_METHODS and resolution is not None:
            resolution_norm = (1.0 / resolution) / (1.0 / 1.2)
            resolution_norm = max(0.0, min(1.0, resolution_norm))
            method_norm = PDB_METHOD_SCORE.get(method, 0.85)
            raw = (
                PDB_WEIGHTS["resolution"] * resolution_norm
                + PDB_WEIGHTS["method"] * method_norm
            )
        elif method in NON_RESOLUTION_METHODS and resolution is None:
            raw = PDB_METHOD_SCORE.get(method, 0.6)
        else:
            print(f"[WARNING] UNKNOWN METHOD: {method} has resolution {resolution}")
            raw = 0.4

        calibrated = raw ** 0.75
        calibrated *= STRUCTURE_DB_MULTIPLIER["PDB"]
        structure.analysis.structural_plausibility = round(calibrated, 4)
        return structure

    def score_swiss_model_structure(self, structure: HomologyStructure):
        raw = 0.0
        raw += SWISS_MODEL_WEIGHTS["gmqe"] * structure.metadata.gmqe
        raw += SWISS_MODEL_WEIGHTS["domain_coverage"] * structure.analysis.domain_coverage
        raw += SWISS_MODEL_WEIGHTS["identity"] * structure.metadata.identity / 100
        raw += SWISS_MODEL_WEIGHTS["similarity"] * structure.metadata.similarity
        raw += SWISS_MODEL_WEIGHTS["avg_local_score"] * structure.metadata.avg_local_score

        if structure.metadata.qmean6_z is not None:
            qmean_norm = max(0.0, min(1.0, 1.0 + structure.metadata.qmean6_z / 6.0))
            raw += SWISS_MODEL_WEIGHTS["qmean6_z"] * qmean_norm

        calibrated = raw ** 1.05
        calibrated *= STRUCTURE_DB_MULTIPLIER["SMR"]
        structure.analysis.structural_plausibility = round(calibrated, 4)
        return structure

    def score_swiss_model_structures(self, structures: list[HomologyStructure]):
        return [self.score_swiss_model_structure(structure) for structure in structures]

    def structure_selection_score(self, structure: Structure, w_structure=0.7, w_pocket=0.3):
        structural = structure.analysis.structural_plausibility

        best_pocket = max((p.binding_site_score for p in structure.analysis.fpockets), default=0.0)

        if structural <= 0.0 or best_pocket <= 0.0:
            return 0.0

        return (
            (structural ** w_structure)
            * (best_pocket ** w_pocket)
            * structure.analysis.true_pocket_certainty
        )

    def feature_structure_enrichment(
        self,
        structure: Structure,
        domain: List[Region],
        active_sites: List[Region],
    ):
        chains = structure.chains

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

            structure.analysis.domain_coverage = max(domain_coverages) if domain_coverages else 0.0
        else:
            structure.analysis.domain_coverage = 0.0

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

        structure.analysis.active_site_coverage = (
            covered_sites / total_sites if total_sites > 0 else 0.0
        )
        return structure

    def find_best_swiss_model_structure(self, structure: Structure, uniprot: str):
        intermediate_json = self.get_swiss_model_intermediate(structure.struct_id)
        structures = self.parse_swiss_model_intermediate_json(intermediate_json, uniprot)
        scored_structures = self.score_swiss_model_structures(structures)
        return max(scored_structures, key=lambda s: s.analysis.structural_plausibility)

    def find_best_alpha_fold_structure(self, structure: Structure, uniprot: str):
        intermediate_json = self.get_alpha_fold_intermediate(structure.struct_id)
        structures = self.parse_alpha_fold_intermediate_json(intermediate_json, uniprot)
        scored_structures = self.score_alpha_fold_structures(structures)
        return max(scored_structures, key=lambda s: s.analysis.structural_plausibility)

    def add_paths(self, uniprot: str, structure: Structure):
        struct_id = structure.struct_id
        base = STRUCTURES_DIR / uniprot / struct_id

        structure.paths = StructurePaths(
            raw_structure_path=base / f"{struct_id}_raw.cif",
            structure_path=base / f"{struct_id}.pdb",
            fpocket_out=base / "fpocket_out",
            fpocket_pockets=base / "fpocket_out" / "pockets",
            fpocket_txt=base / "fpocket_out" / f"{struct_id}_info.txt",
            prank_out=base / "prank_out",
            prank_csv=base / "prank_out" / f"{struct_id}.pdb_predictions.csv",
            probis_out=base / "probis_out",
            probis_srf=base / "probis_out" / f"{struct_id}.srf",
            probis_compare=COMPARISON_DIR / uniprot / struct_id,
            probis_nosql=COMPARISON_DIR / uniprot / struct_id / "probis.nosql",
            probis_results=COMPARISON_DIR / uniprot / struct_id / "results.json",
        )
        return structure

    def metadata_structure_enrichment(
        self,
        structure: Structure,
        domain: List[Region],
        active_sites: List[Region],
        uniprot: str,
    ):
        if structure.database == "SMR":
            intermediate_json = self.get_swiss_model_intermediate(structure.struct_id)
            structures = self.parse_swiss_model_intermediate_json(intermediate_json, uniprot)
            structures = [
                self.feature_structure_enrichment(candidate, domain, active_sites)
                for candidate in structures
            ]
            scored_structures = self.score_swiss_model_structures(structures)
            structure = max(
                scored_structures,
                key=lambda candidate: candidate.analysis.structural_plausibility,
            )
        elif structure.database == "AlphaFoldDB":
            structure = self.find_best_alpha_fold_structure(structure, uniprot)
            structure = self.feature_structure_enrichment(structure, domain, active_sites)
        else:
            structure = self.score_pdb_structure(structure)
            structure = self.feature_structure_enrichment(structure, domain, active_sites)
        structure = self.add_paths(uniprot, structure)
        return structure

    def metadata_structures_enrichment(
        self,
        structures: list[Structure],
        domain: List[Region],
        active_sites: List[Region],
        uniprot: str,
    ):
        return [
            self.metadata_structure_enrichment(s, domain, active_sites, uniprot)
            for s in structures
        ]

    def pdb_specific_enrichment(self, structure):
        struct_id = structure["struct_id"]
        structure["cif"] = f"https://files.rcsb.org/download/{struct_id}.cif"
        return structure

    def pdb_specific_filtration(self, structure):
        method = structure.get("method")
        resolution = structure.get("resolution")
        if method not in {"X-ray", "EM", "Neutron", "NMR"}:
            return False
        elif resolution is not None and resolution > 3.0:
            return False
        return True
