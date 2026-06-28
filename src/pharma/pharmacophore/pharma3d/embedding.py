import logging
from time import perf_counter

from src.domain.pharm import PairwiseModel, PairDiagnostics
from src.pharma.math.gaussian_nll import gaussian_nll_vectorized
from src.pharma.pharmacophore.pharma3d.screening import embedding_pair_distances


logger = logging.getLogger(__name__)


def score_embedding_with_diagnostics(embedding, model: PairwiseModel):
    started = perf_counter()
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

    logger.debug(
        "Scored Pharm3D embedding diagnostics features=%d pairs=%d total_nll=%.6f elapsed_ms=%.3f",
        len(embedding),
        len(observed),
        total,
        (perf_counter() - started) * 1000,
    )
    return total, diagnostics
