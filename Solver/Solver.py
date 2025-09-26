from concurrent.futures import as_completed

import numpy as np
from pathlib import Path
from Data.InputData import input_data
from Solver.Tour import tour
from Solver.Moves import move
from time import time
from Solver import EXECUTOR, AVAILABLE_PROCESSOR_CORES


class Solver:
    def __init__(self, file_name: str):
        self._file_name = file_name
        repo_root = Path(__file__).resolve().parent.parent
        tsp_dir = repo_root / "ALL_tsp"
        selected_file = tsp_dir / file_name
        if not selected_file.is_file():
            raise FileNotFoundError(f"TSP file not found: {selected_file}")
        self._data = input_data(str(selected_file))

    def Solve(self, **kwargs) -> tour:
        method = kwargs.get("method", "chained_LK")
        if method == "nearest_neighbor":
            if "max_time" in kwargs:
                max_time = kwargs.get("max_time")
                return self._nearest_neighbor(max_time)
            else:
                return self._nearest_neighbor()
        elif method == "chained_LK":
            if "max_time" in kwargs:
                max_time = kwargs.get("max_time")
                return self._chained_LK(max_time)
            else:
                return self._chained_LK()

    def _nearest_neighbor(self, max_time: float = float("inf")) -> tour:
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

    def _chained_LK(self, max_time: float = float("inf")) -> tour:
        if max_time <= 0:
            raise ValueError("max_time must be positive or infinity")
        start_time = time()
        print(f"File = {self._file_name}")
        print(f"Stops Count = {self._data.stops_count}")
        print("Solution approach = Chained Lin-Kernighan")
        best_tour = tour(self._data)
        best_tour_time = start_time
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
                    best_tour.set_reach_time(time() - start_time)
                    print(f"New best cost = {best_tour.cost:.2f} at {int(best_tour.reach_time * 1000)} ms")

        return best_tour
