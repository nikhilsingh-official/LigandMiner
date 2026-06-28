import logging
from time import perf_counter

import numpy as np
from src.domain.pharm import PairwiseModel


logger = logging.getLogger(__name__)


def embedding_pair_distances(embedding, model: PairwiseModel):
    started = perf_counter()
    coords = np.array([[f.GetPos().x, f.GetPos().y, f.GetPos().z] for f in embedding])

    diffs = coords[model.idx_a] - coords[model.idx_b]
    distances = np.linalg.norm(diffs, axis=1)
    logger.debug(
        "Computed Pharm3D geometry distances features=%d pairs=%d elapsed_ms=%.3f",
        len(embedding),
        len(distances),
        (perf_counter() - started) * 1000,
    )
    return distances
