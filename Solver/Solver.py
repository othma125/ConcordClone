from Data.InputData import input_data
from Solver.Tour import tour
from Solver.Moves import move
import numpy as np


class Solver:
    def __init__(self, file_name: str):
        self._file_name = file_name
        self._input_data = input_data(file_name)

    def Solve(self, **kwargs) -> tour:
        pass

    def _nearest_neighbor(self) -> tour:
        n = self._input_data.stops_count
        sequence = np.random.permutation(n).astype(int)
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if self._input_data.get_cost(sequence[i - 1], sequence[j]) < self._input_data.get_cost(sequence[i - 1], sequence[i]):
                    # Swap
                    move(i, j).perform(sequence)
