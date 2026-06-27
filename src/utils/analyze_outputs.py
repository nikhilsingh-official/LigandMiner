import json
import math

INPUT = "output.json"

def analyze(output_json):
    summary = []

    for uniprot, data in output_json.items():
        ligands = data.get("ligands", [])
        if not ligands:
            continue

        # pick best ligand by lowest IC50
        best_ligand = min(
            ligands,
            key=lambda l: l["affinity"]
        )

        affinity = best_ligand["affinity"]
        similarity = data["similarity_score"]

        combined_score = similarity * math.log10(1.0 / affinity)

        summary.append({
            "uniprot": uniprot,
            "struct_id": data["struct_id"],
            "similarity_score": similarity,
            "ligand_id": best_ligand["ligand_id"],
            "affinity": affinity,
            "combined_score": combined_score
        })

    # rank by combined score
    summary.sort(key=lambda x: x["combined_score"], reverse=True)
    return summary

if __name__ == "__main__":
    with open(INPUT) as f:
        output_json = json.load(f)

    ranked = analyze(output_json)

    print("\nTOP CANDIDATES (for slides):\n")
    for r in ranked[:10]:
        print(
            f"{r['uniprot']} | "
            f"STRUCT_ID {r['struct_id']} | "
            f"Ligand {r['ligand_id']} | "
            f"IC50 = {r['affinity']} | "
            f"Similarity = {r['similarity_score']:.2f} | "
            f"Combined = {r['combined_score']:.2f}"
        )
