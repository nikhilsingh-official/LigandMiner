from src.domain.pharm import PairwiseModel, PairDiagnostics
from src.pharma.math.gaussian_nll import gaussian_nll_vectorized
from src.pharma.pharmacophore.pharma3d.screening import embedding_pair_distances


def score_embedding_with_diagnostics(embedding, model: PairwiseModel):
    observed = embedding_pair_distances(embedding, model)

    total, z, nll = gaussian_nll_vectorized(
        observed,
        model.means,
        model.model_stds
    )

    diagnostics = PairDiagnostics(
        observed_distances=observed,
        z_scores=z,
        nll_contributions=nll
    )

    return total, diagnostics
