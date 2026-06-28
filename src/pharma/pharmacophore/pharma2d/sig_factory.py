import logging
from time import perf_counter

from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.rdMolChemicalFeatures import MolChemicalFeatureFactory


logger = logging.getLogger(__name__)


def init_sig_factory(feat_factory: MolChemicalFeatureFactory) -> SigFactory:
    started = perf_counter()
    sig_factory = SigFactory(feat_factory, minPointCount=2, maxPointCount=3, trianglePruneBins=False)
    sig_factory.SetBins([(0, 2), (2, 5), (5, 8)])
    sig_factory.Init()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Initialized Pharm2D signature factory min_points=2 max_points=3 bins=%s sig_size=%d elapsed_ms=%.3f",
            ((0, 2), (2, 5), (5, 8)),
            sig_factory.GetSigSize(),
            (perf_counter() - started) * 1000,
        )
    return sig_factory
