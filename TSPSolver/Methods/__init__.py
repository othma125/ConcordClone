from .nearest_neighbor import NearestNeighbor
from .christofides import Christofides
from .chained_LK import ChainedLK
from .concord_wrapper import ConcordeWrapper
from .pyvrp_hgs import pyVRPHGS

registry = {
    "nearest_neighbor": NearestNeighbor,
    "christofides": Christofides,
    "chained_LK": ChainedLK,
    "concord_wrapper": ConcordeWrapper,
    "pyvrp_hgs": pyVRPHGS,
}

__all__ = ["registry"]
