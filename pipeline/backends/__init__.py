"""Model backend registry — LSS only."""

from .base import BEVBackend
from .lss_backend import LSSBackend

BACKENDS = {"lss": LSSBackend}


def get_backend(name: str = "lss") -> BEVBackend:
    """Get backend instance. Only LSS is available."""
    if name not in BACKENDS:
        raise KeyError(f"Unknown backend '{name}'. Available: {list(BACKENDS.keys())}")
    return BACKENDS[name]()


def list_backends():
    return list(BACKENDS.keys())
