import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Tuple

import rdkit
from rdkit import DataStructs

from src.domain.ligand import Ligand
from src.domain.pharm import Pharm2DConfig, Pharm2DResult, ConsensusFingerprintDict, LigandFingerprint, FingerprintDict, \
    Fingerprint
from src.pharma.pharmacophore.pharma2d.consensus import ligands_to_consensus_fp
from src.pharma.pharmacophore.pharma2d.factories import build_pharm2d_factories
from src.pharma.pharmacophore.pharma2d.featurizer import _featurize_ligand
from src.pharma.preparation.ligand_preparation import attach_mols, attach_mol
from src.pharma.utils.hashing import make_single_fp_id, _from_blob, _to_blob


class Pharm2DScreening:

    def __init__(self, config: Pharm2DConfig):
        self.config = config
        self.sig_factory, self.morgan_factory = build_pharm2d_factories(config)
        self.consensus: ConsensusFingerprintDict | None = None
        self.screen_dataset: List[LigandFingerprint] | None = None
        self.logger = logging.getLogger("pharm2d")

        Path(config.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.db = sqlite3.connect(config.db_path)
        self.db.execute("PRAGMA journal_mode=WAL;")
        self.db.execute("PRAGMA synchronous=NORMAL;")

        self._initialize_cache_schema()

    def _initialize_cache_schema(self):
        self.db.execute("""
                        CREATE TABLE IF NOT EXISTS ligand_fingerprint_cache(
                            ligand_id TEXT,
                            config_hash TEXT,
                            pharm2d BLOB,
                            morgan BLOB,
                            PRIMARY KEY(ligand_id, config_hash)
                        );
                        """)
        self.db.commit()

    def set_consensus(self, ligands: List[Ligand]):
        prepped_ligands = attach_mols(ligands)
        self.consensus = ligands_to_consensus_fp(prepped_ligands, self.sig_factory, self.morgan_factory)

    def compare_one_to_many(
        self,
        reference: Ligand,
        candidates: List[Ligand],
    ) -> List[Tuple[Ligand, Pharm2DResult]]:
        self.set_consensus([reference])
        self.set_screen_dataset(candidates)
        return self.compare_many_to_many()

    def compare_one_to_one(
        self,
        reference: Ligand,
        candidate: Ligand,
    ) -> Tuple[Ligand, Pharm2DResult]:
        self.set_consensus([reference])
        result = self._compare(candidate)
        return candidate, result

    def _fp_config_hash(self) -> str:
        config = {
            "pharm2d": {
                "bin_step": self.config.pharm2d_bin_step,
                "min_points": self.config.pharm2d_min_features,
                "max_points": self.config.pharm2d_max_features,
                "sig_size": self.sig_factory.GetSigSize(),
                "fixed_size_conversion": True,  # Ensures fingerprints use fixed size from sig_factory
            },
            "morgan": {
                "radius": self.config.morgan_radius,
                "nBits": self.config.morgan_nbits,
                "useChirality": getattr(self.morgan_factory, "useChirality", False),
            },
            "rdkit_version": rdkit.__version__,
        }
        raw = json.dumps(config, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()

    def _load_fp_from_db(self, ligand_id, config_hash):
        row = self.db.execute(
            """
            SELECT pharm2d, morgan
            FROM ligand_fingerprint_cache
            WHERE ligand_id = ?
              AND config_hash = ?
            """,
            (ligand_id, config_hash),
        ).fetchone()

        if row is None:
            return None

        return _from_blob(row[0]), _from_blob(row[1])

    def _store_fp_to_db(self, ligand_id, config_hash, pharm2d_blob, morgan_blob):
        self.db.execute(
            """
            INSERT OR REPLACE INTO ligand_fingerprint_cache
            VALUES (?, ?, ?, ?)
            """,
            (ligand_id, config_hash, pharm2d_blob, morgan_blob),
        )

    def set_screen_dataset(self, ligands: List[Ligand]):

        prepped = attach_mols(ligands)

        ids = [lig.record.ligand_id for lig in prepped]
        self.logger.info(f"Number of IDs:{len(ids)}, Number of unique IDs: {len(set(ids))}.")

        dataset: List[LigandFingerprint] = []

        for lig in prepped:

            fps = _featurize_ligand(
                lig,
                self.sig_factory,
                self.morgan_factory
            )

            pharm2d_bits = fps.pharm2d.bits
            morgan_bits = fps.morgan.bits

            dataset.append(
                LigandFingerprint(
                    ligand=lig,
                    fps=FingerprintDict(
                        pharm2d=Fingerprint(
                            fp_id=make_single_fp_id(lig.record.ligand_id, "pharm2d"),
                            bits=pharm2d_bits,
                            source_id=lig.record.ligand_id
                        ),
                        morgan=Fingerprint(
                            fp_id=make_single_fp_id(lig.record.ligand_id, "morgan"),
                            bits=morgan_bits,
                            source_id=lig.record.ligand_id
                        ),
                    )
                )
            )

        self.screen_dataset = dataset

    def _compare(self, candidate: Ligand):

        if self.consensus is None:
            raise RuntimeError("Consensus not set. Call set_consensus() first.")
        prepped_candidate = attach_mol(candidate)
        candidate_fingerprints = _featurize_ligand(prepped_candidate, self.sig_factory, self.morgan_factory)

        pharm2d_similarity = DataStructs.TanimotoSimilarity(self.consensus.pharm2d.bits, candidate_fingerprints.pharm2d.bits)
        morgan_similarity = DataStructs.TanimotoSimilarity(self.consensus.morgan.bits, candidate_fingerprints.morgan.bits)

        return Pharm2DResult(pharm2d_similarity, morgan_similarity)

    def compare_many_to_many(self) -> List[Tuple[Ligand, Pharm2DResult]]:
        if self.consensus is None:
            raise RuntimeError("Consensus not set.")
        if self.screen_dataset is None:
            raise RuntimeError("Screen dataset not set.")

        pharm_fps = [x.fps.pharm2d.bits for x in self.screen_dataset]
        morgan_fps = [x.fps.morgan.bits for x in self.screen_dataset]

        pharm_sims = DataStructs.BulkTanimotoSimilarity(
            self.consensus.pharm2d.bits, pharm_fps
        )
        morgan_sims = DataStructs.BulkTanimotoSimilarity(
            self.consensus.morgan.bits, morgan_fps
        )

        results = []
        for fp, p_sim, m_sim in zip(self.screen_dataset, pharm_sims, morgan_sims):
            results.append(
                (
                    fp.ligand,
                    Pharm2DResult(p_sim, m_sim)
                )
            )

        return results
