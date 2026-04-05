import math


def rms(values: list[float]) -> float:
    if not values:
        return float("inf")
    return math.sqrt(sum(v * v for v in values) / len(values))


def normalized_deviations(distances, lowers, uppers):
    deviations = []

    for d, L, U in zip(distances, lowers, uppers):
        halfwidth = (U - L) / 2.0
        if halfwidth == 0:
            continue
        mid = (L + U) / 2.0
        deviations.append((d - mid) / halfwidth)

    return deviations
