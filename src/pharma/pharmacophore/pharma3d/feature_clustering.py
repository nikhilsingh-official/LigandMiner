from typing import List

from rdkit.Chem.rdChemicalFeatures import FreeChemicalFeature
from rdkit.Geometry import Point3D

from src.domain.ligand import LigandPharmacophore
from src.domain.pharm import ClusterFeature, Cluster


def build_feature_clusters(ligands: List[LigandPharmacophore], proximity: float) -> List[Cluster]:

    feature_clusters = []

    for ligand in ligands:
        ligand_id = ligand.record.ligand_id
        pcophore = ligand.pharmacophore

        for f in pcophore.getFeatures():

            added = False

            for cluster in feature_clusters:

                if not is_same_feature(f, cluster):
                    continue

                if not is_near_to_centroid(f, cluster, proximity):
                    continue

                new_centroid, new_stddev, new_M2 = get_updated_params(cluster, f.GetPos())

                cluster.centroid = new_centroid
                cluster.stddev = new_stddev
                cluster.M2 = new_M2

                cluster.features.append(ClusterFeature(
                    ligand_id=ligand_id,
                    feature=f
                ))
                cluster.ligand_ids.add(ligand_id)
                added = True
                break

            if not added:
                feature_clusters.append(
                    Cluster(
                        ligand_ids={ligand_id},
                        features=[ClusterFeature(
                            ligand_id=ligand_id,
                            feature=f
                        )
                        ],
                        centroid=f.GetPos(),
                        family=f.GetFamily(),
                        type=f.GetType(),
                        stddev=0.0,
                        M2=0.0
                    )
                )

    return feature_clusters


def get_updated_params(cluster: Cluster, new_pos: Point3D):
    n = len(cluster.features)

    if n == 0:
        return new_pos, 0.0, 0.0

    old_centroid = cluster.centroid
    old_M2 = cluster.M2

    dx = new_pos.x - old_centroid.x
    dy = new_pos.y - old_centroid.y
    dz = new_pos.z - old_centroid.z

    n_new = n + 1

    new_centroid = Point3D(
        old_centroid.x + dx / n_new,
        old_centroid.y + dy / n_new,
        old_centroid.z + dz / n_new,
    )

    dx2 = new_pos.x - new_centroid.x
    dy2 = new_pos.y - new_centroid.y
    dz2 = new_pos.z - new_centroid.z

    new_M2 = old_M2 + (dx * dx2 + dy * dy2 + dz * dz2)

    variance = new_M2 / n_new
    stddev = variance ** 0.5

    return new_centroid, stddev, new_M2

def is_near_to_centroid(feature: FreeChemicalFeature, cluster: Cluster, proximity=1.5) -> bool:
    return feature.GetPos().Distance(cluster.centroid) < proximity


def is_same_feature(f1: FreeChemicalFeature, cluster: Cluster) -> bool:
    return f1.GetFamily() == cluster.family and f1.GetType() == cluster.type
