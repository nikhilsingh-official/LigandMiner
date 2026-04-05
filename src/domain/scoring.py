from typing import Dict
from src.domain.common import AlphaFoldMetrics, MethodType, PDBMetrics, StructureDB, SwissModelMetrics

PDB_METHOD_SCORE: Dict[MethodType, float] = {
    "X-ray": 1.0,
    "EM": 0.8,
    "Neutron": 0.7,
    "NMR": 0.6,
}

STRUCTURE_DB_MULTIPLIER: Dict[StructureDB, float] = {
    "PDB": 1.0,
    "SMR": 0.8,
    "AlphaFoldDB": 0.6,
}

PDB_WEIGHTS: Dict[PDBMetrics, float] = {
    "resolution": 0.7,
    "method": 0.3
}

SWISS_MODEL_WEIGHTS: Dict[SwissModelMetrics, float] = {
    "gmqe": 0.25,
    "domain_coverage": 0.25,
    "identity": 0.15,
    "similarity": 0.15,
    "avg_local_score": 0.10,
    "qmean6_z": 0.10
}

ALPHA_FOLD_WEIGHTS: Dict[AlphaFoldMetrics, float] = {
    "avg_local_score": 0.70,
    "coverage": 0.30
}