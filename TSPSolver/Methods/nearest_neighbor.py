from TSPSolver.TSPTour import TSPTour
from TSPSolver.TSPSolver import TSPSolver
from time import time
from TSPSolver.Moves import move


class NearestNeighborSolver(TSPSolver):
    """Adapter that subclasses TSPSolver and delegates to run_nearest_neighbor."""

    def __init__(self, file_name: str):
        super().__init__(file_name)

    def solve(self, max_time: float = float("inf")) -> TSPTour:
        """Nearest Neighbor heuristic implementation.

        solver is an instance of TSPSolver; this function operates on its
        internal data (solver._data, solver._file_name).
        """
        if max_time <= 0:
            raise ValueError("max_time must be positive or infinity")
        start_time = time()
        print(f"File = {self._file_name.split('\\')[-1]}")
        print(f"Stops Count = {self._data.stops_count}")
        print("Solution approach = Nearest Neighbor")
        n = self._data.stops_count
        sequence = __import__('numpy').random.permutation(n).astype(int)
        for i in range(1, n - 1):
            if max_time < float("inf") and time() - start_time > max_time:
                break
            x = sequence[i - 1]
            for j in range(i + 1, n):
                if self._data.get_cost(x, sequence[j]) < self._data.get_cost(x, sequence[i]):
                    move(i, j).swap(sequence)
        new_tour = TSPTour(self._data, sequence)
        new_tour.set_reach_time(time() - start_time)
        new_tour.set_solution_methode("Nearest Neighbor")
        print(f"New best cost = {new_tour.cost:.2f} at {int(new_tour.reach_time * 1000)} ms")
        return new_tour
