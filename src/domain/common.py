from typing import Final, Literal, Set

MethodType = Literal["X-ray", "EM", "Neutron", "NMR"]

ResolutionMethodType = Literal["X-ray", "EM", "Neutron"]
NonResolutionMethodType = Literal["NMR"]

RESOLUTION_METHODS: Final[Set[ResolutionMethodType]] = {
    "X-ray",
    "EM",
    "Neutron",
}

NON_RESOLUTION_METHODS: Final[Set[NonResolutionMethodType]] = {
    "NMR",
}

StructureDB = Literal["PDB", "SMR", "AlphaFoldDB"]

AffinityType = Literal["IC50", "Ki", "Kd", "EC50"]

PDBMetrics = Literal["resolution", "method"]
SwissModelMetrics = Literal["gmqe", "domain_coverage", "identity", "similarity", "avg_local_score", "qmean6_z"]
AlphaFoldMetrics = Literal["avg_local_score", "coverage"]