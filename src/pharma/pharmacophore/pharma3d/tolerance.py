import logging
from time import perf_counter

from rdkit.Chem.Pharm3D.Pharmacophore import Pharmacophore


logger = logging.getLogger(__name__)


def create_tolerance(pharmacophore: Pharmacophore, tolerance=1.0):
    started = perf_counter()
    n = len(pharmacophore.getFeatures())

    for i in range(n):
        for j in range(i+1, n):
            lower = pharmacophore.getLowerBound(i, j)
            upper = pharmacophore.getUpperBound(i, j)
            pharmacophore.setLowerBound(i, j, max(0.0, lower - tolerance))
            pharmacophore.setUpperBound(i, j, upper + tolerance)

    logger.debug(
        "Expanded Pharm3D pharmacophore tolerance features=%d pairs=%d tolerance=%.6f elapsed_ms=%.3f",
        n,
        n * (n - 1) // 2,
        tolerance,
        (perf_counter() - started) * 1000,
    )
    return pharmacophore
