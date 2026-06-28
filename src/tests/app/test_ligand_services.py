import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from src.app.ligand_services import LigandService


class LigandDatabaseCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        (root / "resources").mkdir()
        self.db_path = root / "resources" / "ligand.db"
        connection = sqlite3.connect(self.db_path)
        connection.executescript(
            """
            CREATE TABLE ligand (
                ligand_id TEXT PRIMARY KEY,
                inchikey TEXT,
                smiles TEXT
            );
            CREATE TABLE activity (
                protein_id TEXT,
                ligand_id TEXT,
                affinity_type TEXT,
                affinity_value REAL,
                p_affinity REAL
            );
            CREATE TABLE fda_approved_ligands (ligand_id TEXT PRIMARY KEY);
            """
        )
        connection.executemany(
            "INSERT INTO ligand VALUES (?, ?, ?)",
            [
                ("L1", "K1", "CCO"),
                ("L2", "K2", "CCN"),
                ("L3", "K3", "CCC"),
            ],
        )
        connection.executemany(
            "INSERT INTO activity VALUES (?, ?, ?, ?, ?)",
            [
                ("P1", "L1", "IC50", 100.0, 7.0),
                ("P1", "L1", "IC50", 10.0, 8.0),
                ("P1", "L2", "Ki", 50.0, 7.5),
                ("P2", "L1", "Kd", 20.0, 7.8),
                ("P2", "L3", "Other", 1.0, 9.0),
                ("P3", "L3", "EC50", None, 0.0),
            ],
        )
        connection.executemany("INSERT INTO fda_approved_ligands VALUES (?)", [("L1",), ("L2",)])
        connection.commit()
        connection.close()
        self.service = LigandService(session_obj=Mock(), base_dir=root)

    def tearDown(self):
        self.temp.cleanup()


class TestLigandDatabaseQueries(LigandDatabaseCase):
    def test_batch_query_filters_types_and_nulls_and_orders_by_affinity(self):
        records = self.service.get_ligands_for_proteins(["P1", "P2"])

        self.assertEqual(
            [(record.protein_id, record.ligand_id, record.affinity) for record in records],
            [("P1", "L1", 10.0), ("P1", "L2", 50.0), ("P1", "L1", 100.0), ("P2", "L1", 20.0)],
        )

    def test_empty_protein_batch_returns_empty_records(self):
        self.assertEqual(self.service.get_ligands_for_proteins([]), [])

    def test_per_protein_query_keeps_only_strongest_row_per_ligand(self):
        ligands = self.service.fetch_ligands_for_protein("P1")

        self.assertEqual([(item.record.ligand_id, item.record.affinity) for item in ligands], [("L1", 10.0), ("L2", 50.0)])

    def test_unknown_protein_has_no_training_ligands(self):
        self.assertEqual(self.service.fetch_ligands_for_protein("UNKNOWN"), [])

    def test_fda_query_keeps_highest_p_affinity_activity_across_proteins(self):
        records = self.service.get_fda_approved_ligands()
        by_id = {record.ligand_id: record for record in records}

        self.assertEqual(set(by_id), {"L1", "L2"})
        self.assertEqual(by_id["L1"].protein_id, "P1")
        self.assertEqual(by_id["L1"].affinity, 10.0)


class TestChEMBLLigandHandling(unittest.TestCase):
    def test_target_lookup_returns_only_targets_with_matching_accession(self):
        session = Mock()
        session.get.return_value.ok = True
        session.get.return_value.json.return_value = {
            "targets": [
                {"target_chembl_id": "T1", "target_components": [{"accession": "P1"}]},
                {"target_chembl_id": "T2", "target_components": [{"accession": "OTHER"}]},
            ]
        }

        self.assertEqual(LigandService(session).chembl_targets_from_uniprot("P1"), ["T1"])

    def test_activity_parser_skips_invalid_rows_and_keeps_lowest_value(self):
        activities = [
            {"standard_value": "100", "molecule_chembl_id": "L1", "standard_type": "IC50", "canonical_smiles": "CCO"},
            {"standard_value": "10", "molecule_chembl_id": "L1", "standard_type": "IC50", "canonical_smiles": "CCO"},
            {"standard_value": "bad", "molecule_chembl_id": "L2", "standard_type": "Ki", "canonical_smiles": "CCN"},
            {"standard_value": "5", "molecule_chembl_id": "L3", "standard_type": "Ki", "canonical_smiles": None},
        ]

        parsed = LigandService(Mock()).parse_chembl_activities(activities)

        self.assertEqual(parsed["ligand_count"], 1)
        self.assertEqual(parsed["ligands"][0]["affinity"], 10.0)

    def test_merge_deduplicates_by_identity_and_keeps_lowest_affinity(self):
        first = {"ligands": [{"ligand_id": "L1", "smiles": "CCO", "affinity": 20.0}]}
        second = {"ligands": [{"ligand_id": "L1", "smiles": "CCO", "affinity": 5.0}]}

        merged = LigandService(Mock()).merge_ligand_outputs(first, None, second)

        self.assertEqual(merged["ligand_count"], 1)
        self.assertEqual(merged["ligands"][0]["affinity"], 5.0)

    def test_get_ligands_from_chembl_returns_none_without_targets(self):
        session = Mock()
        session.get.return_value.ok = True
        session.get.return_value.json.return_value = {"targets": []}

        self.assertIsNone(LigandService(session).get_ligands_from_chembl("P1"))


if __name__ == "__main__":
    unittest.main()
