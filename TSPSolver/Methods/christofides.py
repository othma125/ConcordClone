from TSPData.TSPInstance import TSPInstance
from TSPSolver.TSPTour import TSPTour
from TSPSolver.Methods.TSPSolver import TSPSolver
from time import time


class ChristofidesSolver(TSPSolver):
    def __init__(self, data: TSPInstance):
        super().__init__(data)

    def solve(self, max_time: float = float("inf")) -> TSPTour:
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
        print(f"File = {self._file_name.split('\\')[-1]}")
        print(f"Stops Count = {self._data.stops_count}")
        print("Solution approach = Christofides")
        n = self._data.stops_count
        graph = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                graph.add_edge(i, j, weight=self._data.get_cost(i, j))
        sequence = __import__('numpy').array(christofides(graph), dtype=int)[:n]
        new_tour = TSPTour(self._data, sequence)
        new_tour.set_reach_time(time() - start_time)
        new_tour.set_solution_methode("Christofides")
        print(f"New best cost = {new_tour.cost:.2f} at {int(new_tour.reach_time * 1000)} ms")
        return new_tour
