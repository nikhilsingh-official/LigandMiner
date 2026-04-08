from __future__ import annotations

from typing import Literal, Optional, List
from pydantic import BaseModel, ConfigDict, Field
from dataclasses import dataclass, field
from typing import List

class UniProtApiModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

EntryType = Literal[
    "UniProtKB reviewed (Swiss-Prot)",
    "UniProtKB unreviewed (TrEMBL)",
]

ProteinExistence = Literal[
    "1: Evidence at protein level",
    "2: Evidence at transcript level",
    "3: Inferred from homology",
    "4: Predicted",
    "5: Uncertain",
]

class EntryAudit(UniProtApiModel):
    firstPublicDate: str
    lastAnnotationUpdateDate: str
    lastSequenceUpdateDate: str
    entryVersion: int
    sequenceVersion: int


class Organism(UniProtApiModel):
    scientificName: str
    commonName: Optional[str] = None
    taxonId: int
    lineage: List[str]


class Sequence(UniProtApiModel):
    value: str
    length: int
    molWeight: int
    crc64: str
    md5: str

class FeatureLocationRange(UniProtApiModel):
    value: Optional[int] = None
    modifier: Optional[str] = None


class FeatureLocation(UniProtApiModel):
    start: FeatureLocationRange
    end: FeatureLocationRange


class Feature(UniProtApiModel):
    type: str
    location: FeatureLocation
    description: Optional[str] = None

class Property(UniProtApiModel):
    key: str
    value: str

class UniProtKBCrossReference(UniProtApiModel):
    database: str
    id: str
    properties: Optional[List[Property]] = None

class ExtraAttributes(UniProtApiModel):
    uniParcId: Optional[str] = None

class UniProtEntry(UniProtApiModel):
    entryType: EntryType
    primaryAccession: str

    features: Optional[List[Feature]] = None
    uniProtKBCrossReferences: Optional[List[UniProtKBCrossReference]] = None

    extraAttributes: ExtraAttributes

@dataclass
class Region:
    start: int
    end: int

type InterProIDs = List[str]
type Domains = List[Region]
type ActiveSites = List[Region]

@dataclass
class SequenceAnnotations:
    interpro_ids: InterProIDs = field(default_factory=list)
    domains: Domains = field(default_factory=list)
    active_sites: ActiveSites = field(default_factory=list)