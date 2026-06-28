from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List

from src.domain.ligand import Ligand, LigandRecord
from src.domain.pharm import Pharm2DConfig, Pharm3DConfig
from src.domain.pipeline import PipelineState
from src.pharma.pharmacophore.pharma2d.screening import Pharm2DScreening
from src.pharma.pharmacophore.pharma3d.screening import Pharm3DScreening

from .context import session
from .progress import NullProgressSink, ProgressSink


VALID_TYPES = {"IC50", "Ki", "Kd", "EC50"}


class ScreeningHit:
    def __init__(self, ligand, pharm2d=None, pharm3d=None, combined_score=None):
        self.ligand = ligand
        self.pharm2d = pharm2d
        self.pharm3d = pharm3d
        self.combined_score = combined_score


class LigandService:
    def __init__(self, session_obj=session, base_dir: Path | None = None):
        self.session = session_obj
        self.base_dir = base_dir or Path(__file__).resolve().parents[2]
        self.db_path = self.base_dir / "resources" / "ligand.db"

    def chembl_targets_from_uniprot(self, uniprot):
        url = "https://www.ebi.ac.uk/chembl/api/data/target/search.json"
        params = {"q": uniprot}

        res = self.session.get(url, params=params, timeout=30)
        if not res.ok:
            return []

        data = res.json()
        targets = []
        for target in data.get("targets", []):
            for comp in target.get("target_components", []):
                if comp.get("accession") == uniprot:
                    targets.append(target["target_chembl_id"])
                    break

        return targets

    def chembl_activities_from_targets(self, target_ids, limit=200):
        activities = []

        for tid in target_ids:
            url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
            params = {
                "target_chembl_id": tid,
                "standard_type__in": ",".join(VALID_TYPES),
                "standard_relation": "=",
                "limit": limit,
            }

            res = self.session.get(url, params=params, timeout=30)
            if not res.ok:
                continue

            activities.extend(res.json().get("activities", []))

        return activities

    def parse_chembl_activities(self, activities):
        ligands = {}

        for act in activities:
            try:
                value = float(act["standard_value"])
                ligand_id = act["molecule_chembl_id"]
                affinity_type = act["standard_type"]
                smiles = act["canonical_smiles"]

                if not smiles:
                    continue
            except (KeyError, TypeError, ValueError):
                continue

            if ligand_id not in ligands or value < ligands[ligand_id]["affinity"]:
                ligands[ligand_id] = {
                    "ligand_id": ligand_id,
                    "smiles": smiles,
                    "affinity_type": affinity_type,
                    "affinity": value,
                }

        return {
            "ligands": list(ligands.values()),
            "ligand_count": len(ligands),
        }

    def get_ligands_from_chembl(self, uniprot: str):
        targets = self.chembl_targets_from_uniprot(uniprot)
        if not targets:
            return None

        activities = self.chembl_activities_from_targets(targets)
        if not activities:
            return None

        return self.parse_chembl_activities(activities)

    def merge_ligand_outputs(self, *outputs):
        merged = {}

        for out in outputs:
            if not out:
                continue

            for lig in out.get("ligands", []):
                dedup_key = (
                    lig.get("inchi_key")
                    or lig.get("ligand_id")
                    or lig.get("smiles")
                )

                if not dedup_key:
                    continue

                if (
                    dedup_key not in merged
                    or lig["affinity"] < merged[dedup_key]["affinity"]
                ):
                    merged[dedup_key] = lig

        return {
            "ligands": list(merged.values()),
            "ligand_count": len(merged),
        }

    def get_ligands_for_proteins(self, uniprot_ids: list[str]) -> List[LigandRecord]:
        placeholders = ",".join(["?"] * len(uniprot_ids))

        query = f"""
    SELECT
        a.protein_id,
        l.inchikey,
        l.ligand_id,
        l.smiles,
        a.affinity_type,
        a.affinity_value
    FROM activity a
    JOIN ligand l ON l.ligand_id = a.ligand_id
    WHERE a.protein_id IN ({placeholders})
      AND a.affinity_value IS NOT NULL
      AND a.affinity_type IN ('Ki','Kd','IC50','EC50')
    ORDER BY a.protein_id, a.p_affinity DESC
    """

        conn = sqlite3.connect(self.db_path)
        cur = conn.execute(query, uniprot_ids)
        rows = cur.fetchall()

        return [
            LigandRecord(
                protein_id=r[0],
                inchi_key=r[1],
                ligand_id=r[2],
                smiles=r[3],
                affinity_type=r[4],
                affinity=float(r[5]),
            )
            for r in rows
        ]

    def get_fda_approved_ligands(self) -> List[LigandRecord]:
        query = """
SELECT
    a.protein_id,
    l.inchikey,
    l.ligand_id,
    l.smiles,
    a.affinity_type,
    a.affinity_value
FROM (
    SELECT
        a2.protein_id,
        a2.ligand_id,
        a2.affinity_type,
        a2.affinity_value,
        ROW_NUMBER() OVER (
            PARTITION BY a2.ligand_id
            ORDER BY a2.p_affinity DESC
        ) as rn
    FROM activity a2
    JOIN fda_approved_ligands fda
        ON fda.ligand_id = a2.ligand_id
    WHERE a2.affinity_value IS NOT NULL
      AND a2.affinity_type IN ('Ki','Kd','IC50','EC50')
) a
JOIN ligand l ON l.ligand_id = a.ligand_id
WHERE a.rn = 1;"""

        conn = sqlite3.connect(self.db_path)
        cur = conn.execute(query)
        rows = cur.fetchall()

        return [
            LigandRecord(
                protein_id=r[0],
                inchi_key=r[1],
                ligand_id=r[2],
                smiles=r[3],
                affinity_type=r[4],
                affinity=float(r[5]),
            )
            for r in rows
        ]

    def fetch_ligands_for_protein(self, uniprot_id: str) -> List[Ligand]:
        query = """
    SELECT DISTINCT
        a.protein_id,
        l.inchikey,
        l.ligand_id,
        l.smiles,
        a.affinity_type,
        a.affinity_value
    FROM (
        SELECT
            a2.protein_id,
            a2.ligand_id,
            a2.affinity_type,
            a2.affinity_value,
            ROW_NUMBER() OVER (
                PARTITION BY a2.protein_id, a2.ligand_id
                ORDER BY a2.p_affinity DESC
            ) as rn
        FROM activity a2
        WHERE a2.protein_id = ?
          AND a2.affinity_value IS NOT NULL
          AND a2.affinity_type IN ('Ki','Kd','IC50','EC50')
    ) a
    JOIN ligand l ON l.ligand_id = a.ligand_id
    WHERE a.rn = 1
    """

        conn = sqlite3.connect(self.db_path)
        cur = conn.execute(query, (uniprot_id,))
        rows = cur.fetchall()
        conn.close()

        ligands = []
        for r in rows:
            record = LigandRecord(
                protein_id=r[0],
                inchi_key=r[1],
                ligand_id=r[2],
                smiles=r[3],
                affinity_type=r[4],
                affinity=float(r[5]),
            )
            ligands.append(Ligand(record=record))

        return ligands


class ScreeningService:
    def __init__(self, ligand_service: LigandService | None = None):
        self.ligand_service = ligand_service or LigandService()

    def select_pharm3d_training_set(self, training_ligands: List[Ligand]):
        if not training_ligands or len(training_ligands) < 2:
            return []

        ligs = sorted(
            training_ligands,
            key=lambda l: (
                l.record.affinity is None,
                l.record.affinity if l.record.affinity is not None else float("inf"),
            ),
        )

        n = len(ligs)
        min_ligands = 10
        percent = int(n * 0.3)
        max_ligands = 30

        k = max(min_ligands, percent)
        k = min(k, max_ligands, n)

        if k < 2:
            return []

        return ligs[:k]

    def screen_ranked_proteins(
        self,
        pipeline: PipelineState,
        fda_approved_ligands: List[Ligand],
        progress: ProgressSink | None = None,
    ) -> list[dict]:
        results = []
        progress = progress or NullProgressSink()

        if not pipeline.ranked_proteins:
            return results

        progress.start(
            "screen_ranked_proteins",
            f"Screening {len(pipeline.ranked_proteins)} proteins...",
            total=len(pipeline.ranked_proteins),
        )

        pharm2d = Pharm2DScreening(Pharm2DConfig())
        pharm3d = Pharm3DScreening(
            Pharm3DConfig(
                feature_consensus_ratio=0.4,
                proximity=2.5,
                pair_distance_support_ratio=0.2,
                pairwise_bounds_k=3.0,
                num_conformers=20,
                energy_window=10.0,
                progress=progress.child("pharm3d"),
            )
        )

        pharm2d.set_screen_dataset(fda_approved_ligands)

        for protein in pipeline.ranked_proteins:
            uniprot_id = protein.uniprot_id
            probis_score = protein.probis_score

            progress.update("screen_ranked_proteins", f"Screening protein {uniprot_id} (fetching training ligands)...")

            training_ligands = self.ligand_service.fetch_ligands_for_protein(uniprot_id)

            if not training_ligands:
                progress.advance("screen_ranked_proteins")
                continue

            num_training = len(training_ligands)

            progress.update("screen_ranked_proteins", f"{uniprot_id}: building Pharm2D consensus ({num_training} ligands)")

            pharm2d.set_consensus(training_ligands)

            progress.update("screen_ranked_proteins", f"{uniprot_id}: Pharm2D screening {len(fda_approved_ligands)} FDA ligands")

            pharm2d_results = pharm2d.compare_many_to_many()

            hits = []
            for lig, pharm2d_result in pharm2d_results:
                combined_score = (
                    0.6 * pharm2d_result.pharm2d_similarity
                    + 0.4 * pharm2d_result.morgan_similarity
                )
                hits.append(
                    ScreeningHit(
                        ligand=lig,
                        pharm2d=pharm2d_result,
                        combined_score=combined_score,
                    )
                )

            hits.sort(key=lambda h: h.combined_score, reverse=True)
            num_to_keep = max(1, int(0.25 * len(hits)))
            hits = hits[:num_to_keep]

            pharm3d_training = self.select_pharm3d_training_set(training_ligands)

            if len(pharm3d_training) < 2:
                progress.update("screen_ranked_proteins", f"{uniprot_id}: skipping Pharm3D (only {len(pharm3d_training)} ligand)")
            else:
                progress.update("screen_ranked_proteins", f"{uniprot_id}: Pharm3D consensus ({len(pharm3d_training)} ligands)")

                pharm3d.set_consensus(pharm3d_training)
                pharm3d.set_screen_dataset([h.ligand for h in hits])

                progress.update("screen_ranked_proteins", f"{uniprot_id}: Pharm3D screening {len(hits)} candidates")

                pharm3d_results = pharm3d.compare_many_to_many()

                pharm3d_by_ligand_id = {
                    lig.record.ligand_id: res
                    for lig, res in pharm3d_results
                }

                for hit in hits:
                    lig_id = hit.ligand.record.ligand_id
                    if lig_id in pharm3d_by_ligand_id:
                        hit.pharm3d = pharm3d_by_ligand_id[lig_id]

            def get_pharm3d_score(hit):
                if hit.pharm3d is None or not hit.pharm3d.embeddings:
                    return float("inf")
                return min(e.score for e in hit.pharm3d.embeddings)

            hits_sorted = sorted(hits, key=get_pharm3d_score)

            hit_records = []
            for hit in hits_sorted:
                pharm3d_score = None
                if hit.pharm3d is not None and hit.pharm3d.embeddings:
                    pharm3d_score = min(e.score for e in hit.pharm3d.embeddings)

                training_affinity_type = None
                training_affinity_value = None
                for train_lig in training_ligands:
                    if train_lig.record.ligand_id == hit.ligand.record.ligand_id:
                        training_affinity_type = train_lig.record.affinity_type
                        training_affinity_value = train_lig.record.affinity
                        break

                hit_records.append(
                    {
                        "ligand_id": hit.ligand.record.ligand_id,
                        "smiles": hit.ligand.record.smiles,
                        "training_affinity_type": training_affinity_type,
                        "training_affinity_value": training_affinity_value,
                        "pharm2d_similarity": hit.pharm2d.pharm2d_similarity,
                        "morgan_similarity": hit.pharm2d.morgan_similarity,
                        "combined_2d_score": hit.combined_score,
                        "pharm3d_score": pharm3d_score,
                    }
                )

            results.append(
                {
                    "protein": uniprot_id,
                    "probis_score": probis_score,
                    "num_training_ligands": num_training,
                    "hits": hit_records,
                }
            )

            progress.advance("screen_ranked_proteins")

        progress.finish("screen_ranked_proteins", "Pharmacophore screening complete")
        pharm2d.close()
        return results
