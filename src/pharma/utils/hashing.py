import hashlib
from rdkit import DataStructs


def make_single_fp_id(ligand_id: int | str, kind: str) -> str:
    digest = hashlib.sha1(str(ligand_id).encode()).hexdigest()[:12]
    return f"{kind}_{digest}"


def make_consensus_fp_id(ligand_ids: list[int | str], kind: str) -> str:
    joined = "|".join(sorted(str(lid) for lid in ligand_ids))
    digest = hashlib.sha1(joined.encode()).hexdigest()[:12]
    return f"{kind}_consensus_{digest}"



def _to_blob(fp):
    return DataStructs.BitVectToBinaryText(fp)

def _from_blob(blob):
    return DataStructs.CreateFromBinaryText(blob)