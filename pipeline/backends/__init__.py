"""Model backend registry."""

from .base import BEVBackend
from .lss_backend import LSSBackend
from .tpvformer_backend import TPVFormerBackend

# Import conditionally to avoid errors if dependencies are missing
try:
    from .gaussianformer_backend import GaussianFormerBackend
except ImportError:
    GaussianFormerBackend = None

try:
    from .sparseocc_backend import SparseOccBackend
except ImportError:
    SparseOccBackend = None

# Registry of all available backends
BACKENDS = {}

def _register(cls):
    if cls is not None:
        BACKENDS[cls.name] = cls

_register(LSSBackend)
_register(TPVFormerBackend)
if GaussianFormerBackend:
    _register(GaussianFormerBackend)
if SparseOccBackend:
    _register(SparseOccBackend)


def get_backend(name: str) -> BEVBackend:
    """Get a backend instance by name."""
    if name not in BACKENDS:
        raise KeyError(f"Unknown backend '{name}'. Available: {list(BACKENDS.keys())}")
    return BACKENDS[name]()


def list_backends():
    """List all registered backend names."""
    return list(BACKENDS.keys())
