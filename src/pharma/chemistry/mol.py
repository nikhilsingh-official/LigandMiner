import logging
from time import perf_counter

from src.pharma.preparation.conformer_generation import generate_filtered_conformers
from src.domain.ligand import LigandMol


logger = logging.getLogger(__name__)

def ligand_to_conformers(
    ligand: LigandMol,
    num_conformers: int = 8,
    energy_window: float = 5.0,
):
    started = perf_counter()
    conformers = generate_filtered_conformers(
        ligand.mol,
        ligand.record.ligand_id,
        num_confs=num_conformers,
        energy_window=energy_window,
    )
    logger.debug(
        "Resolved ligand conformers ligand_id=%s conformers=%d elapsed_ms=%.3f",
        ligand.record.ligand_id,
        len(conformers) if conformers else 0,
        (perf_counter() - started) * 1000,
    )
    return conformers
