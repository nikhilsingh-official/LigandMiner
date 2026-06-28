import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdChemicalFeatures import FreeChemicalFeature
from rdkit.Geometry import Point3D

from src.domain.ligand import Conformer, Ligand, LigandConformers, LigandRecord
from src.domain.pharm import Cluster, ClusterFeature, PairwiseModel


def ligand(smiles: str, ligand_id: str) -> Ligand:
    return Ligand(
        LigandRecord(
            protein_id="test-protein",
            inchi_key=f"key-{ligand_id}",
            ligand_id=str(ligand_id),
            smiles=smiles,
            affinity_type="IC50",
            affinity=100.0,
        )
    )


def prepared_ligand(smiles: str, ligand_id: str) -> LigandConformers:
    record = ligand(smiles, ligand_id).record
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    conf_id = AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
    if conf_id < 0:
        raise RuntimeError(f"could not embed test molecule {smiles}")
    return LigandConformers(
        record=record,
        mol=mol,
        conformers=[Conformer(conf_id=conf_id, convergence_success=True, energy=0.0)],
    )


def feature(
    x: float,
    y: float = 0.0,
    z: float = 0.0,
    family: str = "Acceptor",
    feature_type: str | None = None,
) -> FreeChemicalFeature:
    return FreeChemicalFeature(
        family,
        feature_type or family,
        Point3D(float(x), float(y), float(z)),
    )


def cluster(
    family: str,
    feature_type: str,
    members: list[tuple[str, tuple[float, float, float]]],
) -> Cluster:
    features = [
        ClusterFeature(ligand_id, feature(*coords, family=family, feature_type=feature_type))
        for ligand_id, coords in members
    ]
    coords = np.asarray([coords for _, coords in members], dtype=float)
    centroid = coords.mean(axis=0)
    deviations = coords - centroid
    m2 = float(np.sum(deviations * deviations))
    return Cluster(
        ligand_ids={ligand_id for ligand_id, _ in members},
        features=features,
        centroid=Point3D(*map(float, centroid)),
        stddev=(m2 / len(members)) ** 0.5,
        family=family,
        type=feature_type,
        M2=m2,
    )


def pairwise_model(
    idx_a=(), idx_b=(), means=(), raw_stds=(), model_stds=()
) -> PairwiseModel:
    return PairwiseModel(
        idx_a=np.asarray(idx_a, dtype=int),
        idx_b=np.asarray(idx_b, dtype=int),
        means=np.asarray(means, dtype=float),
        raw_stds=np.asarray(raw_stds, dtype=float),
        model_stds=np.asarray(model_stds, dtype=float),
    )


def point_tuple(point):
    return point.x, point.y, point.z
