"""
ChainedLK implements the Chained Lin-Kernighan heuristic for solving the Traveling Salesman Problem (TSP).
Classes:
    ChainedLK(TSPSolver): Inherits from TSPSolver and applies the Chained Lin-Kernighan heuristic.
Methods:
    __init__(self, data: TSPInstance):
        Initializes the ChainedLK solver with a TSPInstance.
    solve(self, max_time: float = float("inf")) -> TSPTour:
        Runs the Chained Lin-Kernighan heuristic to find a near-optimal TSP tour.
        Args:
            max_time (float, optional): Maximum allowed computation time in seconds. Defaults to infinity.
        Returns:
            TSPTour: The best tour found within the allowed time.
        Raises:
            ValueError: If max_time is not positive or infinity.
        Details:
            - Prints information about the problem instance and solution progress.
            - Uses parallel processing to perform multiple perturbations of the current best tour.
            - Updates the best tour when a better candidate is found.
            - Stops when the allowed stagnation time is exceeded or the maximum time is reached.
"""
from TSPData.TSPInstance import TSPInstance
from TSPSolver.TSPTour import TSPTour
from TSPSolver.Methods.TSPSolver import TSPSolver
from time import time
from concurrent.futures import as_completed
import numpy as np


class ChainedLK(TSPSolver):
    def __init__(self, data: TSPInstance):
        super().__init__(data)

    def solve(self, max_time: float = float("inf")) -> TSPTour:
        """ Chained Lin-Kernighan heuristic for TSP """
        if max_time <= 0:
            raise ValueError("max_time must be positive or infinity")
        from TSPSolver import EXECUTOR, AVAILABLE_PROCESSOR_CORES
        start_time = time()
        print(f"File = {self._file_name.split('\\')[-1]}")
        print(f"Stops Count = {self._data.stops_count}")
        print("Solution approach = Chained Lin-Kernighan")
        best_tour = TSPTour(self._data)
        best_tour_time = time()
        best_tour.set_reach_time(best_tour_time - start_time)
        best_tour.set_method("Chained LK")
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
                    best_tour.set_method("Chained LK")
                    print(f"New best cost = {best_tour.cost:.2f} at {int(best_tour.reach_time * 1000)} ms")
            batch.clear()

        return best_tour
