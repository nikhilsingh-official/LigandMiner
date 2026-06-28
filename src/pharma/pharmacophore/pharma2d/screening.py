import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from time import perf_counter
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


logger = logging.getLogger(__name__)


class Pharm2DScreening:

    def __init__(self, config: Pharm2DConfig):
        started = perf_counter()
        self.config = config
        self.sig_factory, self.morgan_factory = build_pharm2d_factories(config)
        self.consensus: ConsensusFingerprintDict | None = None
        self.screen_dataset: List[LigandFingerprint] | None = None
        self.logger = logger

        logger.info("Initializing Pharm2D screening cache database=%s", config.db_path)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Pharm2D configuration morgan_radius=%d morgan_nbits=%d min_features=%d max_features=%d bin_step=%.6f consensus_threshold=%.6f database=%s",
                config.morgan_radius,
                config.morgan_nbits,
                config.pharm2d_min_features,
                config.pharm2d_max_features,
                config.pharm2d_bin_step,
                config.consensus_threshold,
                config.db_path,
            )

        Path(config.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.db = sqlite3.connect(config.db_path)
        self.db.execute("PRAGMA journal_mode=WAL;")
        self.db.execute("PRAGMA synchronous=NORMAL;")

        self._initialize_cache_schema()
        self.config_hash = self._fp_config_hash()
        logger.debug(
            "Pharm2D screening initialized config_hash=%s database=%s elapsed_ms=%.3f",
            self.config_hash,
            config.db_path,
            (perf_counter() - started) * 1000,
        )


    def _initialize_cache_schema(self):
        started = perf_counter()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Ensuring SQLite cache table ligand_fingerprint_cache exists")
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
        logger.debug("Initialized Pharm2D cache schema and committed elapsed_ms=%.3f", (perf_counter() - started) * 1000)

    def set_consensus(self, ligands: List[Ligand]):
        started = perf_counter()
        logger.info("Building Pharm2D consensus ligands=%d", len(ligands))
        try:
            ligands_with_mols = attach_mols(ligands)
            self.consensus = ligands_to_consensus_fp(ligands_with_mols, self.sig_factory, self.morgan_factory, self.config.consensus_threshold)
        except Exception as exc:
            logger.exception(
                "Pharm2D consensus failed ligands=%d stage=consensus_generation exception_type=%s",
                len(ligands),
                type(exc).__name__,
            )
            raise
        logger.info(
            "Finished Pharm2D consensus ligands=%d elapsed_ms=%.3f",
            len(ligands),
            (perf_counter() - started) * 1000,
        )

    def compare_one_to_many(
        self,
        reference: Ligand,
        candidates: List[Ligand],
    ) -> List[Tuple[Ligand, Pharm2DResult]]:
        logger.info("Starting Pharm2D one-to-many screening candidates=%d", len(candidates))
        self.set_consensus([reference])
        self.set_screen_dataset(candidates)
        return self.compare_many_to_many()

    def compare_one_to_one(
        self,
        reference: Ligand,
        candidate: Ligand,
    ) -> Tuple[Ligand, Pharm2DResult]:
        logger.info("Starting Pharm2D one-to-one screening candidate_id=%s", candidate.record.ligand_id)
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
        config_hash = hashlib.sha256(raw).hexdigest()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Computed Pharm2D fingerprint configuration hash=%s config=%s", config_hash, config)
        return config_hash

    def _load_fp_from_db(self, ligand_id, config_hash):
        started = perf_counter()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Executing SQLite fingerprint cache lookup ligand_id=%s config_hash=%s",
                ligand_id,
                config_hash,
            )
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
            logger.debug(
                "Pharm2D fingerprint cache miss ligand_id=%s config_hash=%s elapsed_ms=%.3f",
                ligand_id,
                config_hash,
                (perf_counter() - started) * 1000,
            )
            return None

        try:
            fingerprints = _from_blob(row[0]), _from_blob(row[1])
        except Exception as exc:
            logger.exception(
                "Corrupt Pharm2D fingerprint cache entry ligand_id=%s config_hash=%s exception_type=%s",
                ligand_id,
                config_hash,
                type(exc).__name__,
            )
            raise
        logger.debug(
            "Pharm2D fingerprint cache hit ligand_id=%s config_hash=%s elapsed_ms=%.3f",
            ligand_id,
            config_hash,
            (perf_counter() - started) * 1000,
        )
        return fingerprints

    def _store_fp_to_db(self, ligand_id, config_hash, pharm2d_blob, morgan_blob):
        started = perf_counter()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Executing SQLite fingerprint cache insert ligand_id=%s config_hash=%s",
                ligand_id,
                config_hash,
            )
        self.db.execute(
            """
            INSERT OR REPLACE INTO ligand_fingerprint_cache
            VALUES (?, ?, ?, ?)
            """,
            (ligand_id, config_hash, pharm2d_blob, morgan_blob),
        )
        logger.debug(
            "Inserted Pharm2D fingerprint cache entry ligand_id=%s config_hash=%s elapsed_ms=%.3f",
            ligand_id,
            config_hash,
            (perf_counter() - started) * 1000,
        )

    def _get_or_compute_fp(self, ligand):
        started = perf_counter()
        ligand_id = ligand.record.ligand_id
        cached = self._load_fp_from_db(
            ligand_id,
            self.config_hash,
        )

        if cached is not None:
            pharm2d_bits, morgan_bits = cached

        else:
            try:
                fps = _featurize_ligand(
                    ligand,
                    self.sig_factory,
                    self.morgan_factory,
                )
            except Exception as exc:
                logger.exception(
                    "Failed to generate Pharm2D fingerprints ligand_id=%s smiles=%s stage=fingerprint_generation exception_type=%s",
                    ligand_id,
                    ligand.record.smiles,
                    type(exc).__name__,
                )
                raise

            pharm2d_bits = fps.pharm2d.bits
            morgan_bits = fps.morgan.bits

            self._store_fp_to_db(
                ligand.record.ligand_id,
                self.config_hash,
                _to_blob(pharm2d_bits),
                _to_blob(morgan_bits),
            )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Resolved Pharm2D fingerprints ligand_id=%s source=%s elapsed_ms=%.3f",
                ligand_id,
                "cache" if cached is not None else "generated",
                (perf_counter() - started) * 1000,
            )

        return FingerprintDict(
            pharm2d=Fingerprint(
                fp_id=make_single_fp_id(
                    ligand.record.ligand_id,
                    "pharm2d"
                ),
                bits=pharm2d_bits,
                source_id=ligand.record.ligand_id,
            ),
            morgan=Fingerprint(
                fp_id=make_single_fp_id(
                    ligand.record.ligand_id,
                    "morgan"
                ),
                bits=morgan_bits,
                source_id=ligand.record.ligand_id,
            ),
        )

    def set_screen_dataset(self, ligands: List[Ligand]):
        started = perf_counter()
        logger.info("Preparing Pharm2D screening dataset ligands=%d", len(ligands))
        try:
            ligands_with_mols = attach_mols(ligands)
        except Exception as exc:
            logger.exception(
                "Failed to prepare Pharm2D screening dataset ligands=%d stage=ligand_preparation exception_type=%s",
                len(ligands),
                type(exc).__name__,
            )
            raise

        ids = [lig.record.ligand_id for lig in ligands_with_mols]
        logger.debug("Pharm2D dataset ligand_ids=%d unique_ligand_ids=%d", len(ids), len(set(ids)))

        dataset: List[LigandFingerprint] = []
        cached_before = self.db.total_changes

        for lig in ligands_with_mols:

            try:
                fps = self._get_or_compute_fp(lig)
            except Exception as exc:
                logger.error(
                    "Pharm2D candidate processing failed ligand_id=%s smiles=%s stage=dataset_fingerprinting exception_type=%s",
                    lig.record.ligand_id,
                    lig.record.smiles,
                    type(exc).__name__,
                )
                raise

            dataset.append(
                LigandFingerprint(
                    ligand=lig,
                    fps=fps,
                )
            )

        self.db.commit()
        inserted = self.db.total_changes - cached_before
        logger.debug("Committed Pharm2D fingerprint cache inserts=%d", inserted)

        self.screen_dataset = dataset
        logger.info(
            "Prepared Pharm2D screening dataset ligands=%d loaded_cached=%d generated_or_updated=%d elapsed_ms=%.3f",
            len(dataset),
            len(dataset) - inserted,
            inserted,
            (perf_counter() - started) * 1000,
        )

    def _compare(self, candidate: Ligand):
        started = perf_counter()
        if self.consensus is None:
            logger.error("Cannot screen Pharm2D candidate_id=%s because consensus is not set", candidate.record.ligand_id)
            raise RuntimeError("Consensus not set. Call set_consensus() first.")

        prepped_candidate = attach_mol(candidate)
        candidate_fingerprints = self._get_or_compute_fp(prepped_candidate)

        pharm2d_similarity = DataStructs.TanimotoSimilarity(self.consensus.pharm2d.bits, candidate_fingerprints.pharm2d.bits)
        morgan_similarity = DataStructs.TanimotoSimilarity(self.consensus.morgan.bits, candidate_fingerprints.morgan.bits)

        logger.debug(
            "Scored Pharm2D candidate_id=%s pharm2d_similarity=%.6f morgan_similarity=%.6f elapsed_ms=%.3f",
            candidate.record.ligand_id,
            pharm2d_similarity,
            morgan_similarity,
            (perf_counter() - started) * 1000,
        )

        return Pharm2DResult(pharm2d_similarity, morgan_similarity)

    def compare_many_to_many(self) -> List[Tuple[Ligand, Pharm2DResult]]:
        started = perf_counter()
        if self.consensus is None:
            logger.error("Cannot start Pharm2D screening because consensus is not set")
            raise RuntimeError("Consensus not set.")
        if self.screen_dataset is None:
            logger.error("Cannot start Pharm2D screening because screen dataset is not set")
            raise RuntimeError("Screen dataset not set.")

        logger.info("Screening %d ligands with Pharm2D", len(self.screen_dataset))

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

        elapsed_ms = (perf_counter() - started) * 1000
        if results:
            average_pharm2d = sum(result.pharm2d_similarity for _, result in results) / len(results)
            average_morgan = sum(result.morgan_similarity for _, result in results) / len(results)
            top_ligand, top_result = max(
                results,
                key=lambda item: item[1].pharm2d_similarity + item[1].morgan_similarity,
            )
            logger.info(
                "Finished Pharm2D screening ligands=%d average_pharm2d_similarity=%.6f average_morgan_similarity=%.6f top_ligand_id=%s top_pharm2d_similarity=%.6f top_morgan_similarity=%.6f elapsed_ms=%.3f",
                len(results),
                average_pharm2d,
                average_morgan,
                top_ligand.record.ligand_id,
                top_result.pharm2d_similarity,
                top_result.morgan_similarity,
                elapsed_ms,
            )
        else:
            logger.info("Finished Pharm2D screening ligands=0 elapsed_ms=%.3f", elapsed_ms)

        return results

    def close(self):
        try:
            self.db.commit()
            logger.debug("Committed Pharm2D fingerprint cache during close database=%s", self.config.db_path)
            self.db.close()
            logger.debug("Closed Pharm2D fingerprint cache database=%s", self.config.db_path)
        except Exception:
            logger.exception("Failed to close Pharm2D fingerprint cache database=%s", self.config.db_path)
            pass
