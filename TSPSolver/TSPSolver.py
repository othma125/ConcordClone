from concurrent.futures import as_completed

import numpy as np
from pathlib import Path
from TSPData.TSPInstance import TSPInstance
from TSPSolver.Tour import tour
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

    def Solve(self, **kwargs) -> tour:
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

    def _nearest_neighbor(self, max_time: float = float("inf")) -> tour:
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
        new_tour = tour(self._data, sequence)
        new_tour.set_reach_time(time() - start_time)
        print(f"New best cost = {new_tour.cost:.2f} at {int(new_tour.reach_time * 1000)} ms")
        return new_tour

    def _christofides(self) -> tour:
        """ Christofides heuristic for TSP using NetworkX """
        import networkx as nx
        from networkx.algorithms.approximation import christofides
        start_time = time()
        print(f"File = {self._file_name}")
        print(f"Stops Count = {self._data.stops_count}")
        print("Solution approach = Christofides")
        n = self._data.stops_count
        G = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(i, j, weight=self._data.get_cost(i, j))
        cycle = christofides(G)
        sequence = np.array(cycle, dtype=int)[:n]
        new_tour = tour(self._data, sequence)
        new_tour.set_reach_time(time() - start_time)
        print(f"New best cost = {new_tour.cost:.2f} at {int(new_tour.reach_time * 1000)} ms")
        return new_tour

    def _chained_LK(self, max_time: float = float("inf")) -> tour:
        """ Chained Lin-Kernighan heuristic for TSP """
        if max_time <= 0:
            raise ValueError("max_time must be positive or infinity")
        from TSPSolver import EXECUTOR, AVAILABLE_PROCESSOR_CORES
        start_time = time()
        print(f"File = {self._file_name}")
        print(f"Stops Count = {self._data.stops_count}")
        print("Solution approach = Chained Lin-Kernighan")
        best_tour = tour(self._data)
        best_tour_time = time()
        best_tour.set_reach_time(best_tour_time - start_time)
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
                    print(f"New best cost = {best_tour.cost:.2f} at {int(best_tour.reach_time * 1000)} ms")

        return best_tour

    def _concord_wrapper(self, max_time: float = float("inf"), heuristic: bool = True) -> tour:
        if max_time <= 0:
            raise ValueError("max_time must be positive or infinity")
        from pytsp.tsp import TSPSolver as concord
        start_time = time()
        n = self._data.stops_count
        if n > 10000:
            raise ValueError("Concorde wrapper supports up to 10,000 stops due to memory constraints.")
        print(f"File = {self._file_name}")
        print(f"Stops Count = {n}")
        print("Solution approach = Concord TSP TSPSolver")

        if self._data.matrix is not None:
            solver = concord.from_data(
                n, self._data.matrix, norm="EXPLICIT", sym=True, heuristic=heuristic
            )
        else:
            repo_root = Path(__file__).resolve().parent.parent
            tsp_dir = repo_root / "TSPLIB"
            selected_file = tsp_dir / self._file_name
            solver = concord.from_tspfile(selected_file, heuristic=heuristic)
        solution = solver.solve()
        new_tour = tour(self._data, np.array(solution.tour, dtype=int))
        new_tour.set_reach_time(time() - start_time)
        print(f"New best cost = {new_tour.cost:.2f} at {int(new_tour.reach_time * 1000)} ms")
        return new_tour

    def __del__(self) -> None:
        del self._data
