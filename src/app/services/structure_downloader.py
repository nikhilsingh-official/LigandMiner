from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.domain.pipeline import HasStructures
from src.domain.structure import Structure

from ..context import session
from ..progress import ProgressSink


class StructureDownloader:
    def __init__(self, session_obj=session):
        self.session = session_obj

    def format_chains(self, structure):
        chain_ids = sorted(c.chain for c in structure.chains)
        return ",".join(f"{c}" for c in chain_ids)

    def clean_structure(self, input_cif, output_pdb, structure):
        input_path = Path(input_cif)
        output_path = Path(output_pdb)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        selection = self.format_chains(structure)

        cmd = [
            "gemmi", "convert",
            str(input_path), str(output_path),
            "--select", selection,
            "--remove-lig-wat",
            "--remove-h",
            "--anisou=no",
            "--minimal",
            "--shorten",
        ]

        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def download_structure(self, structure: Structure, progress: ProgressSink, task_key: str, root):
        struct_id = structure.struct_id.upper()
        progress.update(task_key, f"Downloading structures ({struct_id})...")

        structure.paths.raw_structure_path.parent.mkdir(parents=True, exist_ok=True)

        if structure.paths.raw_structure_path.exists():
            progress.advance(task_key)
            return structure

        res = self.session.get(structure.cif)

        if res.status_code != 200:
            raise RuntimeError(f"Failed to download {struct_id} (HTTP {res.status_code})")

        with open(structure.paths.raw_structure_path, "wb") as f:
            f.write(res.content)

        progress.advance(task_key)
        return structure

    def download_structures(
        self,
        protein: HasStructures,
        progress: ProgressSink,
        task_key: str,
        root="./",
        max_workers=6,
    ):
        downloaded = []
        root = Path(root)
        structures = protein.structures.structures

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    self.download_structure,
                    structure,
                    progress,
                    task_key,
                    root,
                ): structure
                for structure in structures
            }

            for future in as_completed(futures):
                structure = futures[future]
                struct_id = structure.struct_id
                try:
                    result = future.result()
                    if result is not None:
                        downloaded.append(result)
                except Exception as e:
                    print(f"[DOWNLOAD FAIL] {struct_id}: {e}")

        return downloaded
