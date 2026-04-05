import numpy as np
from src.domain.pharm import PairwiseModel


def embedding_pair_distances(embedding, model: PairwiseModel):
    coords = np.array([[f.GetPos().x, f.GetPos().y, f.GetPos().z] for f in embedding])

    diffs = coords[model.idx_a] - coords[model.idx_b]
    return np.linalg.norm(diffs, axis=1)
