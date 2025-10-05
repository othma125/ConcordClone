from TSPSolver.TSPTour import TSPTour
from TSPSolver.TSPSolver import TSPSolver
from time import time
from concurrent.futures import as_completed
import numpy as np


class ChainedLKSolver(TSPSolver):
    def __init__(self, file_name: str):
        super().__init__(file_name)

    def solve(self, max_time: float = float("inf")) -> TSPTour:
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
