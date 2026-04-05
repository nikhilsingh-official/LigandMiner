from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.rdMolChemicalFeatures import MolChemicalFeatureFactory


def init_sig_factory(feat_factory: MolChemicalFeatureFactory) -> SigFactory:
    sig_factory = SigFactory(feat_factory, minPointCount=2, maxPointCount=3, trianglePruneBins=False)
    sig_factory.SetBins([(0, 2), (2, 5), (5, 8)])
    sig_factory.Init()
    return sig_factory