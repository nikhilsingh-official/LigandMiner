import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.app.progress import NullProgressSink
from src.app.services.structure_downloader import StructureDownloader
from src.domain.pipeline import ProteinData, ProteinStructures
from src.domain.structure import Chain, ExperimentalStructure, StructurePaths


def structure(struct_id, root):
    return ExperimentalStructure(
        struct_id,
        "PDB",
        chains=[Chain("B", 1, 10), Chain("A", 1, 10)],
        cif=f"https://example/{struct_id}.cif",
        paths=StructurePaths(
            raw_structure_path=root / f"{struct_id}_raw.cif",
            structure_path=root / f"{struct_id}.pdb",
        ),
    )


class TestStructureDownloader(unittest.TestCase):
    def test_chain_format_is_sorted_and_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            item = structure("1ABC", Path(tmp))
            self.assertEqual(StructureDownloader(Mock()).format_chains(item), "A,B")

    def test_existing_download_is_idempotently_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            item = structure("1ABC", Path(tmp))
            item.paths.raw_structure_path.write_bytes(b"existing")
            session = Mock()

            returned = StructureDownloader(session).download_structure(
                item, NullProgressSink(), "download", Path(tmp)
            )

            self.assertIs(returned, item)
            self.assertEqual(item.paths.raw_structure_path.read_bytes(), b"existing")
            session.get.assert_not_called()

    def test_download_writes_response_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            item = structure("1ABC", Path(tmp))
            session = Mock()
            session.get.return_value.status_code = 200
            session.get.return_value.content = b"data"

            StructureDownloader(session).download_structure(
                item, NullProgressSink(), "download", Path(tmp)
            )

            self.assertEqual(item.paths.raw_structure_path.read_bytes(), b"data")

    def test_failed_download_raises_with_structure_and_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            item = structure("1ABC", Path(tmp))
            session = Mock()
            session.get.return_value.status_code = 404

            with self.assertRaisesRegex(RuntimeError, "1ABC.*404"):
                StructureDownloader(session).download_structure(
                    item, NullProgressSink(), "download", Path(tmp)
                )

    def test_clean_structure_invokes_gemmi_with_selected_chains(self):
        with tempfile.TemporaryDirectory() as tmp, patch("subprocess.run") as run:
            item = structure("1ABC", Path(tmp))

            StructureDownloader(Mock()).clean_structure(
                item.paths.raw_structure_path, item.paths.structure_path, item
            )

            command = run.call_args.args[0]
            self.assertEqual(command[:2], ["gemmi", "convert"])
            self.assertEqual(command[command.index("--select") + 1], "A,B")
            self.assertTrue(run.call_args.kwargs["check"])

    def test_batch_download_returns_successes_when_one_download_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = structure("GOOD", root)
            bad = structure("BAD", root)
            session = Mock()

            def get(url):
                result = Mock()
                result.status_code = 404 if "BAD" in url else 200
                result.content = b"ok"
                return result

            session.get.side_effect = get
            protein = ProteinData("P1", "UPI", [], structures=ProteinStructures([good, bad]))

            downloaded = StructureDownloader(session).download_structures(
                protein, NullProgressSink(), "download", max_workers=2
            )

            self.assertEqual([item.struct_id for item in downloaded], ["GOOD"])


if __name__ == "__main__":
    unittest.main()
