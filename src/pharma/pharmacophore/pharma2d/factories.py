from src.domain.pharm import Pharm2DConfig
from src.pharma.fingerprints.morgan import init_morgan_factory
from src.pharma.pharmacophore.common.feature_factory import init_feature_factory
from src.pharma.pharmacophore.pharma2d.sig_factory import init_sig_factory


def build_pharm2d_factories(cfg: Pharm2DConfig):
    feat_factory = init_feature_factory()
    morgan_factory = init_morgan_factory(
        radius=cfg.morgan_radius
    )
    sig_factory = init_sig_factory(
        feat_factory
    )
    return sig_factory, morgan_factory
