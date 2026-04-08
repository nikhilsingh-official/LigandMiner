from dataclasses import dataclass, field
from typing import Protocol

from src.domain.ligand import Ligand
from src.domain.structure import Structure
from src.domain.uniprot import SequenceAnnotations


@dataclass
class ProteinIdentity:
    uniprot_id: str
    uniparc_id: str
    other_accessions: list[str]


@dataclass
class ProteinAnnotations:
    annotations: SequenceAnnotations


@dataclass
class ProteinStructures:
    structures: list[Structure] = field(default_factory=list)
    best_structure: Structure | None = None


@dataclass
class ProteinLigands:
    ligands: list[Ligand] = field(default_factory=list)
    ligand_count: int = 0


@dataclass
class ProteinScores:
    probis_score: float | None = None


class HasIdentity(Protocol):
    @property
    def uniprot_id(self) -> str: ...

    @property
    def uniparc_id(self) -> str: ...

    @property
    def other_accessions(self) -> list[str]: ...


class HasAnnotations(Protocol):
    annotations: ProteinAnnotations


class HasStructures(Protocol):
    structures: ProteinStructures


class HasLigands(Protocol):
    ligands: ProteinLigands


class HasScores(Protocol):
    scores: ProteinScores


class HasMetadata(HasIdentity, HasAnnotations, HasStructures, Protocol):
    pass


class HasIdentifiedStructures(HasIdentity, HasStructures, Protocol):
    pass


class HasAnnotatedStructures(HasMetadata, Protocol):
    pass


class HasStructuredLigands(HasIdentity, HasStructures, HasLigands, Protocol):
    pass


class HasScoredProtein(HasIdentity, HasScores, Protocol):
    pass


class HasStructuredScores(HasIdentity, HasStructures, HasScores, Protocol):
    pass


@dataclass(init=False)
class ProteinData:
    identity: ProteinIdentity
    annotations: ProteinAnnotations | None
    structures: ProteinStructures | None
    ligands: ProteinLigands | None
    scores: ProteinScores | None

    def __init__(
        self,
        uniprot_id: str,
        uniparc_id: str,
        other_accessions: list[str],
        *,
        annotations: ProteinAnnotations | None = None,
        structures: ProteinStructures | None = None,
        ligands: ProteinLigands | None = None,
        scores: ProteinScores | None = None,
    ) -> None:
        self.identity = ProteinIdentity(
            uniprot_id=uniprot_id,
            uniparc_id=uniparc_id,
            other_accessions=other_accessions,
        )
        self.annotations = annotations
        self.structures = structures
        self.ligands = ligands
        self.scores = scores

    @property
    def uniprot_id(self) -> str:
        return self.identity.uniprot_id

    @property
    def uniparc_id(self) -> str:
        return self.identity.uniparc_id

    @property
    def other_accessions(self) -> list[str]:
        return self.identity.other_accessions

    @property
    def best_structure(self) -> Structure | None:
        return None if self.structures is None else self.structures.best_structure

    @best_structure.setter
    def best_structure(self, value: Structure | None) -> None:
        if self.structures is None:
            self.structures = ProteinStructures(best_structure=value)
            return

        self.structures.best_structure = value

    @property
    def probis_score(self) -> float | None:
        return None if self.scores is None else self.scores.probis_score

    @probis_score.setter
    def probis_score(self, value: float | None) -> None:
        if self.scores is None:
            self.scores = ProteinScores(probis_score=value)
            return

        self.scores.probis_score = value

    @property
    def ligand_count(self) -> int:
        return 0 if self.ligands is None else self.ligands.ligand_count

    @ligand_count.setter
    def ligand_count(self, value: int) -> None:
        if self.ligands is None:
            self.ligands = ProteinLigands(ligand_count=value)
            return

        self.ligands.ligand_count = value


@dataclass
class PipelineState:
    target_protein: ProteinData
    interpro_group_id: str | None = None
    candidate_proteins: list[ProteinData] = field(default_factory=list)
    filtered_proteins: list[ProteinData] = field(default_factory=list)
    ranked_proteins: list[ProteinData] = field(default_factory=list)
    ranked_ligands: list[Ligand] = field(default_factory=list)
