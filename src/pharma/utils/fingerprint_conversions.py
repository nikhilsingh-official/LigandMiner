from rdkit.DataStructs.cDataStructs import ExplicitBitVect, SparseBitVect
from typing import Optional

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

    return explicit_fp