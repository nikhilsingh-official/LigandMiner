from rdkit.Chem.Pharm3D.Pharmacophore import Pharmacophore


def create_tolerance(pharmacophore: Pharmacophore, tolerance=1.0):
    n = len(pharmacophore.getFeatures())

    for i in range(n):
        for j in range(i+1, n):
            lower = pharmacophore.getLowerBound(i, j)
            upper = pharmacophore.getUpperBound(i, j)
            pharmacophore.setLowerBound(i, j, max(0.0, lower - tolerance))
            pharmacophore.setUpperBound(i, j, upper + tolerance)

    return pharmacophore
