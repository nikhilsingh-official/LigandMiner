import json
import re
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Entry:
    protein: str
    ligand_id: int
    smiles: str
    pharm2d: float
    morgan: float


ligand_pattern = re.compile(
    r"ligand_id=(\d+), .*?smiles='([^']+)'"
)

result_pattern = re.compile(
    r"pharm2d_similarity=([0-9.]+), morgan_similarity=([0-9.]+)"
)


def load_entries(path: str):
    with open(path) as f:
        raw = json.load(f)

    entries = []

    for protein_block in raw:
        protein = protein_block["protein"]

        for ligand_text, result_text in protein_block["results"]:
            lig_match = ligand_pattern.search(ligand_text)
            res_match = result_pattern.search(result_text)

            entries.append(
                Entry(
                    protein=protein,
                    ligand_id=int(lig_match.group(1)),
                    smiles=lig_match.group(2),
                    pharm2d=float(res_match.group(1)),
                    morgan=float(res_match.group(2)),
                )
            )

    return entries


def main():
    entries = load_entries("pharm2d_output.json")

    by_protein = defaultdict(list)
    for e in entries:
        by_protein[e.protein].append(e)

    print("\n=== BEST PER PROTEIN (Pharm2D) ===")
    for protein, lst in by_protein.items():
        best = max(lst, key=lambda x: x.pharm2d)
        print(protein, best.pharm2d, best.ligand_id)

    print("\n=== BEST PER PROTEIN (Morgan) ===")
    for protein, lst in by_protein.items():
        best = max(lst, key=lambda x: x.morgan)
        print(protein, best.morgan, best.ligand_id)

    print("\n=== WORST PER PROTEIN (Pharm2D) ===")
    for protein, lst in by_protein.items():
        worst = min(lst, key=lambda x: x.pharm2d)
        print(protein, worst.pharm2d, worst.ligand_id)

    print("\n=== WORST PER PROTEIN (Morgan) ===")
    for protein, lst in by_protein.items():
        worst = min(lst, key=lambda x: x.morgan)
        print(protein, worst.morgan, worst.ligand_id)

    print("\n=== GLOBAL TOP 10 Pharm2D ===")
    for e in sorted(entries, key=lambda x: x.pharm2d, reverse=True)[:10]:
        print(e.protein, e.ligand_id, e.pharm2d)

    print("\n=== GLOBAL TOP 10 Morgan ===")
    for e in sorted(entries, key=lambda x: x.morgan, reverse=True)[:10]:
        print(e.protein, e.ligand_id, e.morgan)


if __name__ == "__main__":
    main()