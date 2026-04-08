from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.domain.structure import Structure

from .progress import NullProgressSink, ProgressSink


class SurfaceService:
    def probis_similarity_score(self, parsed):
        aligned = parsed["aligned_vertices"]
        sva = parsed["sva"]
        align_score = parsed["alignment_score"]

        sva_term = 1.0 / (1.0 + sva)

        base_score = 0.6 * aligned + 0.25 * align_score + 0.15 * sva_term

        pocket = parsed.get("best_pocket_match")

        if pocket is None:
            return base_score * 0.5

        pocket_overlap = pocket["overlap_fraction"]
        alignment_in_pocket = pocket["alignment_in_pocket_fraction"]
        binding_score = pocket["binding_site_score"]

        pocket_term = (
            0.5 * pocket_overlap
            + 0.3 * alignment_in_pocket
            + 0.2 * binding_score
        )

        return base_score * (1.0 + pocket_term)

    def parse_aligned_pair(self, pair_str):
        left, right = pair_str.split("=")
        _, t_resnum, t_chain = left.split(":")
        _, s_resnum, s_chain = right.split(":")

        return (
            (t_chain, int(t_resnum), " "),
            (s_chain, int(s_resnum), " "),
        )

    def parse_results(
        self,
        target: Structure,
        similar: Structure,
        progress: ProgressSink | None = None,
    ):
        progress = progress or NullProgressSink()
        progress.start("parse_results", f"Parsing results for {similar.struct_id}...", total=1)

        with open(similar.paths.probis_results) as f:
            results = json.load(f)

        if not results:
            print("[SKIP] Not enough similarity, results.json is empty.")
            return None

        alignments = results[0]["alignment"]
        best = max(alignments, key=lambda a: a["scores"]["aligned_vertices"])
        scores = best["scores"]

        aligned_target = set()
        aligned_similar = set()

        for entry in best["aligned_residues"]:
            t_res, s_res = self.parse_aligned_pair(entry["a"])
            aligned_target.add(t_res)
            aligned_similar.add(s_res)

        pocket_matches = []

        for pocket in target.analysis.fpockets:
            pocket_res = set(pocket.residues)
            if not pocket_res:
                continue

            overlap = aligned_target & pocket_res
            if not overlap:
                continue

            overlap_fraction = len(overlap) / len(pocket_res)
            coverage = len(overlap) / len(aligned_target)

            post_score = (
                0.45 * overlap_fraction
                + 0.35 * coverage
                + 0.20 * pocket.binding_site_score
            )

            pocket_matches.append(
                {
                    "pocket_id": pocket.pocket_id,
                    "post_score": post_score,
                    "overlap_count": len(overlap),
                    "overlap_fraction": overlap_fraction,
                    "alignment_in_pocket_fraction": coverage,
                    "binding_site_score": pocket.binding_site_score,
                }
            )

        best_pocket_match = (
            max(pocket_matches, key=lambda x: x["post_score"])
            if pocket_matches
            else None
        )

        progress.advance("parse_results")
        progress.finish("parse_results", f"Parsed results for {similar.struct_id}")

        return {
            "aligned_vertices": scores["aligned_vertices"],
            "sva": scores["sva"],
            "rmsd": scores["rmsd"],
            "alignment_score": scores["alignment_score"],
            "z_score": scores["z_score"],
            "aligned_target_residues": aligned_target,
            "aligned_similar_residues": aligned_similar,
            "best_pocket_match": best_pocket_match,
            "all_pocket_matches": pocket_matches,
        }

    def manage_results(
        self,
        target_structure: Structure,
        similar_structure: Structure,
        progress: ProgressSink | None = None,
    ):
        progress = progress or NullProgressSink()
        self.get_results(target_structure, similar_structure, progress.child("fetch"))
        return self.parse_results(target_structure, similar_structure, progress.child("parse"))

    def surface_extraction(self, structure: Structure, progress: ProgressSink | None = None):
        progress = progress or NullProgressSink()
        progress.start("surface_extraction", f"Extracting surface for {structure.struct_id}...", total=1)

        chains = sorted(c.chain for c in structure.chains)
        chains_selection = ",".join(f"{c}" for c in chains)

        Path(structure.paths.probis_out).mkdir(parents=True, exist_ok=True)

        if (
            structure.paths.probis_srf
            and structure.paths.probis_srf.exists()
            and structure.paths.probis_srf.stat().st_size > 0
        ):
            progress.advance("surface_extraction")
            progress.finish("surface_extraction", f"Surface already available for {structure.struct_id}")
            return structure.paths.probis_srf

        if not structure.paths.structure_path.exists():
            raise FileNotFoundError(structure.paths.structure_path)

        cmd = [
            "./probis",
            "-extract",
            "-f1", str(structure.paths.structure_path),
            "-c1", chains_selection,
            "-srffile", str(structure.paths.probis_srf),
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        progress.advance("surface_extraction")
        progress.finish("surface_extraction", f"Extracted surface for {structure.struct_id}")

    def compare_surfaces(
        self,
        target_structure: Structure,
        similar_structure: Structure,
        progress: ProgressSink | None = None,
    ):
        progress = progress or NullProgressSink()
        progress.start(
            "compare_surfaces",
            f"Comparing {target_structure.struct_id} with {similar_structure.struct_id}...",
            total=1,
        )

        if not target_structure.paths.probis_srf.exists():
            raise FileNotFoundError(target_structure.paths.probis_srf)
        if not similar_structure.paths.probis_srf.exists():
            raise FileNotFoundError(similar_structure.paths.probis_srf)

        target_chains = sorted(c.chain for c in target_structure.chains)
        c1_selection = ",".join(f"{c}" for c in target_chains)

        similar_chains = sorted(c.chain for c in similar_structure.chains)
        c2_selection = ",".join(f"{c}" for c in similar_chains)

        Path(similar_structure.paths.probis_compare).mkdir(parents=True, exist_ok=True)

        nosql_path = similar_structure.paths.probis_nosql
        if nosql_path.exists():
            progress.advance("compare_surfaces")
            progress.finish(
                "compare_surfaces",
                f"Comparison already available for {target_structure.struct_id} and {similar_structure.struct_id}",
            )
            return

        cmd = [
            "./probis",
            "-compare",
            "-f1", str(target_structure.paths.probis_srf),
            "-c1", c1_selection,
            "-f2", str(similar_structure.paths.probis_srf),
            "-c2", c2_selection,
            "-out", str(similar_structure.paths.probis_compare),
            "-param", "parameters.imp",
            "-nosql", "probis.nosql",
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        progress.advance("compare_surfaces")
        progress.finish(
            "compare_surfaces",
            f"Compared {target_structure.struct_id} with {similar_structure.struct_id}",
        )

    def get_results(
        self,
        target_structure: Structure,
        similar_structure: Structure,
        progress: ProgressSink | None = None,
    ):
        progress = progress or NullProgressSink()
        progress.start("get_results", f"Fetching results for {similar_structure.struct_id}...", total=1)

        if not target_structure.paths.structure_path.exists():
            raise FileNotFoundError(target_structure.paths.structure_path)
        if not similar_structure.paths.structure_path.exists():
            raise FileNotFoundError(similar_structure.paths.structure_path)

        target_chains = sorted(c.chain for c in target_structure.chains)
        c1_selection = ",".join(f"{c}" for c in target_chains)

        similar_chains = sorted(c.chain for c in similar_structure.chains)
        c2_selection = ",".join(f"{c}" for c in similar_chains)

        if similar_structure.paths.probis_results and similar_structure.paths.probis_results.exists():
            progress.advance("get_results")
            progress.finish("get_results", f"Results already available for {similar_structure.struct_id}")
            return

        cmd = [
            "./probis",
            "-results",
            "-f1", str(target_structure.paths.structure_path),
            "-c1", c1_selection,
            "-f2", str(similar_structure.paths.structure_path),
            "-c2", c2_selection,
            "-out", str(similar_structure.paths.probis_compare),
            "-nosql", str(similar_structure.paths.probis_nosql),
            "-json", "results.json",
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        progress.advance("get_results")
        progress.finish("get_results", f"Fetched results for {similar_structure.struct_id}")
