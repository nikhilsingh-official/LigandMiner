from typing import List
import math
import numpy as np
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

def fps_to_consensus(
    fps: List[ExplicitBitVect],
    threshold_ratio: float = 0.8
) -> ExplicitBitVect:

    if not fps:
        raise ValueError("No fingerprints provided for consensus generation.")

    if not (0 < threshold_ratio <= 1):
        raise ValueError("Threshold ratio must be between 0 and 1.")

    n_bits = fps[0].GetNumBits()

    for fp in fps[1:]:
        if fp.GetNumBits() != n_bits:
            raise ValueError("All fingerprints must have identical bit length.")

    n_fps = len(fps)
    threshold = math.ceil(threshold_ratio * n_fps)

    bit_matrix = np.zeros((n_fps, n_bits), dtype=np.uint8)

    for i, fp in enumerate(fps):
        ConvertToNumpyArray(fp, bit_matrix[i])

    bit_counts = bit_matrix.sum(axis=0)
    consensus_mask = bit_counts >= threshold

    consensus = ExplicitBitVect(n_bits)
    on_bits = np.where(consensus_mask)[0]

    for bit in on_bits:
        consensus.SetBit(int(bit))

    return consensus