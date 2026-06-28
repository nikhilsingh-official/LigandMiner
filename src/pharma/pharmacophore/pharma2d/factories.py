import logging
from time import perf_counter

from src.domain.pharm import Pharm2DConfig
from src.pharma.fingerprints.morgan import init_morgan_factory
from src.pharma.pharmacophore.common.feature_factory import init_feature_factory
from src.pharma.pharmacophore.pharma2d.sig_factory import init_sig_factory


logger = logging.getLogger(__name__)


def build_pharm2d_factories(cfg: Pharm2DConfig):
    started = perf_counter()
    feat_factory = init_feature_factory()
    morgan_factory = init_morgan_factory(
        radius=cfg.morgan_radius
    )
    sig_factory = init_sig_factory(
        feat_factory
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Built Pharm2D factories sig_size=%d morgan_radius=%d requested_morgan_nbits=%d elapsed_ms=%.3f",
            sig_factory.GetSigSize(),
            cfg.morgan_radius,
            cfg.morgan_nbits,
            (perf_counter() - started) * 1000,
        )
    return sig_factory, morgan_factory
