from concurrent.futures import as_completed

import math
import numpy as np
from pathlib import Path
from TSPData.TSPInstance import TSPInstance
from TSPSolver.TSPTour import TSPTour
from time import time


class TSPSolver:
    def __init__(self, file_name: str):
        self._file_name = file_name
        repo_root = Path(__file__).resolve().parent.parent
        tsp_dir = repo_root / "TSPLIB"
        selected_file = tsp_dir / file_name
        if not selected_file.is_file():
            raise FileNotFoundError(f"TSP file not found: {selected_file}")
        self._data = TSPInstance(str(selected_file))

    def Solve(self, **kwargs) -> TSPTour:
        method = kwargs.get("method", "chained_LK")
        if method == "nearest_neighbor":
            if "max_time" in kwargs:
                max_time = kwargs.get("max_time")
                return self._nearest_neighbor(max_time)
            else:
                return self._nearest_neighbor()
        elif method == "christofides":
            return self._christofides()
        # elif method == "Simulated_Annealing":
        #     if "max_time" in kwargs:
        #         max_time = kwargs.get("max_time")
        #         return self._Simulated_Annealing(max_time)
        #     else:
        #         return self._Simulated_Annealing()
        elif method == "chained_LK":
            if "max_time" in kwargs:
                max_time = kwargs.get("max_time")
                return self._chained_LK(max_time)
            else:
                return self._chained_LK()
        elif method == "concord_wrapper":
            if "max_time" in kwargs:
                max_time = kwargs.get("max_time")
                return self._concord_wrapper(max_time)
            else:
                return self._concord_wrapper()
        elif method == "pyvrp_hgs":
            # Hybrid Genetic Search via pyvrp
            if "max_time" in kwargs:
                max_time = kwargs.get("max_time")
                return self._pyvrp_hgs(max_time)
            else:
                return self._pyvrp_hgs()

    def _nearest_neighbor(self, max_time: float = float("inf")) -> TSPTour:
        """ Nearest Neighbor heuristic for TSP """
        if max_time <= 0:
            raise ValueError("max_time must be positive or infinity")
        from TSPSolver.Moves import move
        start_time = time()
        print(f"File = {self._file_name}")
        print(f"Stops Count = {self._data.stops_count}")
        print("Solution approach = Nearest Neighbor")
        n = self._data.stops_count
        sequence = np.random.permutation(n).astype(int)
        for i in range(1, n - 1):
            if max_time < float("inf") and time() - start_time > max_time:
                break
            x = sequence[i - 1]
            for j in range(i + 1, n):
                if self._data.get_cost(x, sequence[j]) < self._data.get_cost(x, sequence[i]):
                    # Swap
                    move(i, j).swap(sequence)
        new_tour = TSPTour(self._data, sequence)
        new_tour.set_reach_time(time() - start_time)
        new_tour.set_solution_methode("Nearest Neighbor")
        print(f"New best cost = {new_tour.cost:.2f} at {int(new_tour.reach_time * 1000)} ms")
        return new_tour

    def _christofides(self) -> TSPTour:
        """ Christofides heuristic for TSP using NetworkX """
        try:
            import networkx as nx
            from networkx.algorithms.approximation import christofides
        except Exception as e:
            raise ImportError(
                "networkx is required for the 'christofides' solver. Install with `pip install networkx` "
                f"or more info see https://networkx.org. Original error: {e}"
            )
        start_time = time()
        print(f"File = {self._file_name}")
        print(f"Stops Count = {self._data.stops_count}")
        print("Solution approach = Christofides")
        n = self._data.stops_count
        graph = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                graph.add_edge(i, j, weight=self._data.get_cost(i, j))
        sequence = np.array(christofides(graph), dtype=int)[:n]
        new_tour = TSPTour(self._data, sequence)
        new_tour.set_reach_time(time() - start_time)
        new_tour.set_solution_methode("Christofides")
        print(f"New best cost = {new_tour.cost:.2f} at {int(new_tour.reach_time * 1000)} ms")
        return new_tour

    def _chained_LK(self, max_time: float = float("inf")) -> TSPTour:
        """ Chained Lin-Kernighan heuristic for TSP """
        if max_time <= 0:
            raise ValueError("max_time must be positive or infinity")
        from TSPSolver import EXECUTOR, AVAILABLE_PROCESSOR_CORES
        start_time = time()
        print(f"File = {self._file_name}")
        print(f"Stops Count = {self._data.stops_count}")
        print("Solution approach = Chained Lin-Kernighan")
        best_tour = TSPTour(self._data)
        best_tour_time = time()
        best_tour.set_reach_time(best_tour_time - start_time)
        best_tour.set_solution_methode("Chained LK")
        print(f"New best cost = {best_tour.cost:.2f} at {int(best_tour.reach_time * 1000)} ms")
        stagnation_allowed_time = int(max(100, 100 * np.log(self._data.stops_count)))  # ms

        def non_stop_condition(stag_ms: int, start_ms: float, best_ms: float) -> bool:
            if max_time < float("inf"):
                return time() - start_ms < max_time
            current_time = time()
            numerator = current_time - best_ms
            denominator = max(1e-9, current_time - start_ms)
            probability = numerator / denominator
            return (current_time - best_ms) < (stag_ms / 1000) or np.random.random() > probability

        while non_stop_condition(stagnation_allowed_time, start_time, best_tour_time):
            batch = [EXECUTOR.submit(best_tour.perturbation, self._data) for _ in range(AVAILABLE_PROCESSOR_CORES)]
            for fut in as_completed(batch):
                candidate = fut.result()
                if candidate.cost < best_tour.cost:
                    best_tour = candidate
                    best_tour_time = time()
                    best_tour.set_reach_time(best_tour_time - start_time)
                    best_tour.set_solution_methode("Chained LK")
                    print(f"New best cost = {best_tour.cost:.2f} at {int(best_tour.reach_time * 1000)} ms")

        return best_tour

    def _pyvrp_hgs(self, max_time: float = float("inf")) -> TSPTour:
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
        print(f"File = {self._file_name}")
        print(f"Stops Count = {n}")
        print("Solution approach = pyvrp HGS")

        # Build explicit matrix from get_cost (integer distances)
        matrix = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            for j in range(n):
                matrix[i, j] = int(self._data.get_cost(i, j))

        # For pyvrp ProblemData: create 1 depot and n-1 clients (locations = depots + clients)
        depots = [pyvrp.Depot(x=0, y=0)]
        # clients correspond to nodes 1..n-1 in ProblemData indexing
        clients = [pyvrp.Client(x=0, y=0, delivery=[0], pickup=[0]) for _ in range(max(0, n - 1))]

        vehicle_types = [
            pyvrp.VehicleType(num_available=1, capacity=[sum([0] * n) + 1], start_depot=0, end_depot=0)
        ]

        pdata = pyvrp.ProblemData(
            clients=clients,
            depots=depots,
            vehicle_types=vehicle_types,
            distance_matrices=[matrix],
            duration_matrices=[matrix],
            groups=[],
        )

        # Stopping criterion
        stop = pyvrp.stop.MaxRuntime(max_time) if max_time < float("inf") else pyvrp.stop.MaxIterations(100)

        result = pyvrp.solve(pdata, stop)
        solution = result.best
        routes = solution.routes()
        if len(routes) == 0:
            raise RuntimeError("pyvrp returned no routes")

        route = routes[0]
        route_nodes = [int(v) for v in route.visits()]

        # strip depot occurrences (assume depot index 0)
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

        sequence = np.array(normalized, dtype=int)
        new_tour = TSPTour(self._data, sequence)
        new_tour.set_reach_time(time() - start_time)
        new_tour.set_solution_methode("pyvrp HGS")
        print(f"New best cost = {new_tour.cost:.2f} at {int(new_tour.reach_time * 1000)} ms")
        return new_tour

    def _concord_wrapper(self, max_time: float = float("inf")) -> TSPTour:
        """Run Concorde via the pyconcorde wrapper.

        This function imports the `pyconcorde` package from the Python path.
        If you cloned the `pyconcorde` repository locally (for example into
        a sibling folder or into the project), Python will import that local
        package automatically when running from the repository root.

        Recommended ways to make the package available:
        - Clone locally and install in editable/development mode (recommended
          when you want to inspect or modify pyconcorde):

            git clone https://github.com/jvkersch/pyconcorde.git /path/to/pyconcorde
            pip install -e /path/to/pyconcorde

        - Or install directly from GitHub (non-editable):

            pip install "pyconcorde @ git+https://github.com/jvkersch/pyconcorde"

        Note: building pyconcorde on native Windows may fail because the
        package tries to build Concorde/QSopt; using WSL/Linux or a container
        is the most reliable approach if you need the full Concorde binary.
        """
        if max_time <= 0:
            raise ValueError("max_time must be positive or infinity")
        try:
            # Import pyconcorde (will pick up a local cloned package if present)
            from pyconcorde.concorde.tsp import TSPSolver as ConcordeSolver
            from pyconcorde.concorde.util import write_tsp_file
        except Exception as e:
            # Common failure: local source present but compiled Concorde extension
            # (concorde._concorde) is missing on Windows. Detect that and fall
            # back to the pyvrp_hgs solver so the script can continue.
            msg = str(e)
            if isinstance(e, ModuleNotFoundError) or "concorde" in msg:
                print(
                    "Warning: pyconcorde package was found but the compiled Concorde"
                    " extension is missing (this is common on Windows). Falling back to"
                    " 'pyvrp_hgs' solver. To use Concorde, install/build Concorde (or"
                    " run in WSL/Linux) and install pyconcorde. Original error:\n",
                    e,
                )
                try:
                    return self._pyvrp_hgs(max_time)
                except Exception:
                    # If pyvrp fallback also fails, raise the original import error
                    raise ImportError(
                        "pyconcorde could not be imported and pyvrp fallback failed."
                        f" Original import error: {e}"
                    )
            raise ImportError(
                "pyconcorde could not be imported. If you cloned the project locally,"
                " make sure it is on the Python path or installed in editable mode.\n"
                f"Original import error: {e}"
            )

        start_time = time()
        n = self._data.stops_count
        if n > 10000:
            raise ValueError("Concorde wrapper supports up to 10,000 stops due to memory constraints.")
        print(f"File = {self._file_name}")
        print(f"Stops Count = {n}")
        print("Solution approach = Concord TSP (pyconcorde)")
        
        repo_root = Path(__file__).resolve().parent.parent
        tsp_dir = repo_root / "TSPLIB"
        selected_file = tsp_dir / self._file_name
        if not selected_file.is_file():
            raise FileNotFoundError(f"TSP file not found: {selected_file}")

        # Construct solver directly from the TSPLIB file and pass heuristic/time bound
        solver = ConcordeSolver.from_tspfile(str(selected_file))
        time_bound = -1 if max_time == float('inf') else float(max_time)
        comp = solver.solve(time_bound=time_bound, verbose=False, random_seed=0)

        # comp.cycle is a sequence of 1-based node indices
        cycle = np.array(comp.tour, dtype=int) - 1
        # Ensure length matches n (Concorde sometimes omits final return if -1 marker)
        if len(cycle) == n + 1 and cycle[-1] == -1:
            cycle = cycle[:-1]
        if len(cycle) != n:
            raise RuntimeError(f"Concorde returned cycle of length {len(cycle)} for n={n}")

        new_tour = TSPTour(self._data, cycle)
        new_tour.set_reach_time(time() - start_time)
        new_tour.set_solution_methode("Concorde via pyconcorde")
        print(f"New best cost = {new_tour.cost:.2f} at {int(new_tour.reach_time * 1000)} ms")
        return new_tour

    def __del__(self) -> None:
        del self._data

    def Draw(self, tour: TSPTour) -> None:
        """ Draw the tour using networkX, if the coordinates are available """
        if self._data.stops_count > 20:
            print("Drawing skipped: too many stops (>20) to visualize clearly.")
            return
        try:
            coords = self._data.coordinates
            try:
                import matplotlib.pyplot as plt
                import networkx as nx
            except Exception as e:
                raise ImportError(
                    "matplotlib and networkx are required for the 'Draw' method. Install with "
                    f"`pip install matplotlib networkx` or more info see https://matplotlib.org and https://networkx.org. "
                    f"Original error: {e}"
                )
            if not coords:
                raise ValueError("No coordinates available to draw the tour.")
            if len(coords) != self._data.stops_count:
                raise ValueError("Number of coordinates does not match number of stops.")
            n = self._data.stops_count
            graph = nx.DiGraph()
            pos = {}
            labels = {}
            for i in range(n):
                graph.add_node(i)
                loc = coords[i]
                px, py = loc.latitude, loc.longitude
                pos[i] = (px, py)
                labels[i] = str(i + 1)
            seq = tour.sequence
            for i in range(n):
                graph.add_edge(seq[i], seq[(i + 1) % n])
            plt.figure(figsize=(8, 8))
            nx.draw(graph, pos, with_labels=True, labels=labels, node_size=300, node_color='lightblue', font_size=10, font_weight='bold', arrowsize=15)
            plt.title(f"TSP Tour (cost = {tour.cost:.2f})")
            plt.show()
        except Exception as e:
            raise RuntimeError(f"Error drawing TSP tour: {e}")
