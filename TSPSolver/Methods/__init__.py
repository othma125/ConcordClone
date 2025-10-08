from .nearest_neighbor import NearestNeighbor
from .christofides import Christofides
from .chained_LK import ChainedLK
from .concord_wrapper import ConcordeWrapper
from .pyvrp_hgs import pyVRPHGS
from .simulated_annealing import SimulatedAnnealing
# from .concord_clone import ConcordClone

registry = {
    "nearest_neighbor": NearestNeighbor,
    "christofides": Christofides,
    "chained_LK": ChainedLK,
    "concord_wrapper": ConcordeWrapper,
    "pyvrp_hgs": pyVRPHGS,
    "simulated_annealing": SimulatedAnnealing,
    # "concord_clone": ConcordClone,
}

__all__ = ["registry"]
