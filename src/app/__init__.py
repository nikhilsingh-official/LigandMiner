from .context import progress


__all__ = ["LigandMinerApplication", "progress"]


def __getattr__(name):
    if name == "LigandMinerApplication":
        from .application import LigandMinerApplication

        return LigandMinerApplication
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
