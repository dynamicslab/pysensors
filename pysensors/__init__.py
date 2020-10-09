from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass


from .pysensors import SensorSelector

__all__ = [
    # Modules:
    "basis",
    "classification",
    "optimizers",
    "utils",
    # Non-modules:
    "SensorSelector",
]
