import os

from rdkit import RDConfig
from rdkit.Chem.rdMolChemicalFeatures import MolChemicalFeatureFactory, BuildFeatureFactory


def init_feature_factory() -> MolChemicalFeatureFactory:
    fdef = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    feat_factory = BuildFeatureFactory(fdef)
    return feat_factory
