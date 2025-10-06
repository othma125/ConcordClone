from TSPData.TSPInstance import TSPInstance
from TSPSolver.TSPTour import TSPTour
from TSPSolver.Methods.TSPSolver import TSPSolver
from time import time
from concurrent.futures import as_completed
import numpy as np


class SimulatedAnnealing(TSPSolver):
    def __init__(self, data: TSPInstance):
        super().__init__(data)

    def solve(self, max_time: float = float("inf")) -> TSPTour:
        """ Simulated Annealing heuristic for TSP """
        if max_time <= 0:
            raise ValueError("max_time must be positive or infinity")
        from TSPSolver import EXECUTOR, AVAILABLE_PROCESSOR_CORES
        start_time = time()
        print(f"File = {self._file_name.split('\\')[-1]}")
        print(f"Stops Count = {self._data.stops_count}")
        print("Solution approach = Simulated Annealing")
        # simulated annealing parameters
        initial_temp = 1000
        cooling_rate = 0.995
        current_temp = initial_temp
        best_tour = TSPTour(self._data)
        best_tour_time = time()
        best_tour.set_reach_time(best_tour_time - start_time)
        best_tour.set_method("Simulated Annealing")
        print(f"Initial best cost = {best_tour.cost:.2f} at {int(best_tour.reach_time * 1000)} ms")

        def non_stop_condition(stag_ms: int, start_ms: float, best_ms: float) -> bool:
            if max_time < float("inf"):
                return time() - start_ms < max_time
            current_time = time()
            numerator = current_time - best_ms
            denominator = max(1e-9, current_time - start_ms)
            probability = numerator / denominator
            return (current_time - best_ms) < (stag_ms / 1000) or np.random.random() > probability
        
        while non_stop_condition(1000, start_time, best_tour_time):
            batch = [EXECUTOR.submit(best_tour.perturbation, self._data) for _ in range(AVAILABLE_PROCESSOR_CORES)]
            for fut in as_completed(batch):
                candidate = fut.result()
                cost_diff = candidate.cost - best_tour.cost
                if cost_diff < 0 or np.random.rand() < np.exp(-cost_diff / current_temp):
                    if candidate.cost < best_tour.cost:
                        best_tour = candidate
                        best_tour_time = time()
                        best_tour.set_reach_time(best_tour_time - start_time)
                        best_tour.set_method("Simulated Annealing")
                        print(f"New best cost = {best_tour.cost:.2f} at {int(best_tour.reach_time * 1000)} ms")
            batch.clear()
            current_temp *= cooling_rate

        return best_tour
