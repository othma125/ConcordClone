from .nearest_neighbor import NearestNeighborSolver
from .christofides import ChristofidesSolver
from .chained_LK import ChainedLKSolver
from .concord_wrapper import ConcordeWrapperSolver
from .pyvrp_hgs import PyvrpHGSSolver

registry = {
    "nearest_neighbor": NearestNeighborSolver,
    "christofides": ChristofidesSolver,
    "chained_LK": ChainedLKSolver,
    "concord_wrapper": ConcordeWrapperSolver,
    "pyvrp_hgs": PyvrpHGSSolver,
}

__all__ = ["registry"]
