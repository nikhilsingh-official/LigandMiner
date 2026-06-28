import logging
import math
from time import perf_counter
from typing import List

import numpy as np
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.DataStructs.cDataStructs import ExplicitBitVect


logger = logging.getLogger(__name__)

def fps_to_consensus(
    fps: List[ExplicitBitVect],
    threshold_ratio: float = 0.8
) -> ExplicitBitVect:
    started = perf_counter()

    if not fps:
        logger.warning("Cannot generate fingerprint consensus from an empty fingerprint set")
        raise ValueError("No fingerprints provided for consensus generation.")

    if not (0 < threshold_ratio <= 1):
        logger.error("Invalid fingerprint consensus threshold_ratio=%r", threshold_ratio)
        raise ValueError("Threshold ratio must be between 0 and 1.")

    n_bits = fps[0].GetNumBits()

    for fp in fps[1:]:
        if fp.GetNumBits() != n_bits:
            logger.error("Fingerprint dimension mismatch expected_bits=%d actual_bits=%d", n_bits, fp.GetNumBits())
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

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Generated fingerprint consensus fingerprints=%d dimensions=%d threshold_ratio=%.6f threshold_count=%d on_bits=%d elapsed_ms=%.3f",
            n_fps,
            n_bits,
            threshold_ratio,
            threshold,
            len(on_bits),
            (perf_counter() - started) * 1000,
        )

    return consensus
