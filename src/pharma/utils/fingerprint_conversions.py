import logging
from rdkit.DataStructs.cDataStructs import ExplicitBitVect, SparseBitVect
from typing import Optional


logger = logging.getLogger(__name__)

def sparse_to_explicit(sparse_fp: SparseBitVect, fixed_size: Optional[int] = None) -> ExplicitBitVect:
    """
    Convert a SparseBitVect to an ExplicitBitVect.

    Args:
        sparse_fp: The sparse fingerprint to convert
        fixed_size: If provided, use this as the fingerprint size instead of sparse_fp.GetNumBits().
                   This ensures all fingerprints have the same length.
    """
    n_bits = fixed_size if fixed_size is not None else sparse_fp.GetNumBits()
    explicit_fp = ExplicitBitVect(n_bits)

    for bit in sparse_fp.GetOnBits():
        explicit_fp.SetBit(bit)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Converted sparse fingerprint source_bits=%d target_bits=%d on_bits=%d fixed_size=%s",
            sparse_fp.GetNumBits(),
            n_bits,
            explicit_fp.GetNumOnBits(),
            fixed_size,
        )

    return explicit_fp
