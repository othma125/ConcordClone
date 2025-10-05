from TSPData.TSPInstance import TSPInstance
from TSPSolver.TSPTour import TSPTour
from TSPSolver.Methods.TSPSolver import TSPSolver
from time import time
import numpy as _np


class PyvrpHGSSolver(TSPSolver):
    def __init__(self, data: TSPInstance):
        super().__init__(data)

    def solve(self, max_time: float = float("inf")) -> TSPTour:
        """Use pyvrp's Hybrid Genetic Search (HGS) to solve the TSP instance.

        This method performs a lazy import of pyvrp and translates the current
        TSPInstance into pyvrp's expected format. If pyvrp is not installed,
        a clear ImportError is raised with installation instructions.
        Install it with pip (this will also try to fetch/build VROOM):
            pip install pyvrp or pip install "pyvrp @ git+https://github.com/VROOM-Project/pyvrp"
        """
        try:
            import pyvrp
        except Exception as e:
            raise ImportError(
                "pyvrp is required for the 'pyvrp_hgs' solver. Install with `pip install pyvrp` "
                f"or more info see https://github.com/VROOM-Project/pyvrp. Original error: {e}"
            )

        start_time = time()
        n = self._data.stops_count
        print(f"File = {self._file_name.split('\\')[-1]}")
        print(f"Stops Count = {n}")
        print("Solution approach = pyvrp HGS")

        # Build explicit matrix from get_cost (integer distances)
        matrix = _np.zeros((n, n), dtype=_np.int64)
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i, j] = int(self._data.get_cost(i, j))
                matrix[j, i] = matrix[i, j]

        depots = [pyvrp.Depot(x=0, y=0)]
        clients = [pyvrp.Client(x=0, y=0, delivery=[0], pickup=[0]) for _ in range(max(0, n - 1))]
        vehicle_types = [pyvrp.VehicleType(num_available=1, capacity=[1], start_depot=0, end_depot=0)]

        pdata = pyvrp.ProblemData(
            clients=clients,
            depots=depots,
            vehicle_types=vehicle_types,
            distance_matrices=[matrix],
            duration_matrices=[matrix],
            groups=[],
        )

        stop = pyvrp.stop.MaxRuntime(max_time) if max_time < float("inf") else pyvrp.stop.MaxIterations(100)

        result = pyvrp.solve(pdata, stop)
        solution = result.best
        routes = solution.routes()
        if len(routes) == 0:
            raise RuntimeError("pyvrp returned no routes")

        route = routes[0]
        route_nodes = [int(v) for v in route.visits()]

        while route_nodes and route_nodes[0] == 0:
            route_nodes.pop(0)
        while route_nodes and route_nodes[-1] == 0:
            route_nodes.pop()

        normalized = None
        if len(route_nodes) == n and set(route_nodes) == set(range(n)):
            normalized = route_nodes
        if normalized is None and len(route_nodes) == n - 1 and set(route_nodes) == set(range(1, n)):
            normalized = [0] + [v for v in route_nodes]
        if normalized is None:
            nodes_no_depot = [v for v in route_nodes if v != 0]
            if len(nodes_no_depot) == n and set(nodes_no_depot) == set(range(n)):
                normalized = nodes_no_depot
            elif len(nodes_no_depot) == n - 1 and set(nodes_no_depot) == set(range(1, n)):
                normalized = [0] + nodes_no_depot
        if normalized is None:
            max_node = max(route_nodes) if route_nodes else -1
            if max_node == n:
                try_map = [v - 1 for v in route_nodes]
                if len(try_map) == n and set(try_map) == set(range(n)):
                    normalized = try_map
        if normalized is None:
            raise RuntimeError(f"Unable to normalize pyvrp route to TSP permutation. n={n}, route={route_nodes}")

        sequence = _np.array(normalized, dtype=int)
        new_tour = TSPTour(self._data, sequence)
        new_tour.set_reach_time(time() - start_time)
        new_tour.set_solution_methode("pyvrp HGS")
        print(f"New best cost = {new_tour.cost:.2f} at {int(new_tour.reach_time * 1000)} ms")
        return new_tour
