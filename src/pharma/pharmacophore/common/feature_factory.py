import logging
import os
from time import perf_counter

from rdkit import RDConfig
from rdkit.Chem.rdMolChemicalFeatures import MolChemicalFeatureFactory, BuildFeatureFactory


logger = logging.getLogger(__name__)


def init_feature_factory() -> MolChemicalFeatureFactory:
    started = perf_counter()
    fdef = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    feat_factory = BuildFeatureFactory(fdef)
    logger.debug(
        "Initialized RDKit chemical feature factory definition=%s elapsed_ms=%.3f",
        fdef,
        (perf_counter() - started) * 1000,
    )
    return feat_factory
