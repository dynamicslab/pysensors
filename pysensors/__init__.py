from importlib.metadata import PackageNotFoundError, distribution

try:
    __version__ = distribution("python-sensors").version
except PackageNotFoundError:
    pass

from .classification import SSPOC
from .reconstruction import SSPOR

__all__ = [
    # Modules:
    "basis",
    "classification",
    "reconstruction",
    "optimizers",
    "utils",
    # Non-modules:
    "SSPOR",
    "SSPOC",
]
