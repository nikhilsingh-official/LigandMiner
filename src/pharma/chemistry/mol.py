from src.pharma.preparation.conformer_generation import generate_filtered_conformers
from src.domain.ligand import LigandMol

def ligand_to_conformers(ligand: LigandMol):
    return generate_filtered_conformers(
        ligand.mol,
        ligand.record.ligand_id,
        num_confs=8,
        energy_window=5.0
    )