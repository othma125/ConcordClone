from Solver.Moves import Move
from Data.InputData import InputData
from .LocalSearchMoves import LocalSearchMove

class RightShiftMove(LocalSearchMove):
    """Right-shift move that moves the element at index j to index i, shifting others left."""

    def __init__(self, sequence, i, j, degree: int = 1, withTwoOpt: bool = True) -> None:
        super().__init__(sequence, i, j)
        self._degree = degree
        self._withTwoOpt = withTwoOpt

    def set_gain(self, data: InputData) -> None:
        n = len(self._sequence)
        # Trivial case and gain computation
        if self.i - self._degree == self.j or (self.i - self._degree == 0 and self.j + 1 == n):
            self._gain = 0.0
            return
        # borders management
        x = n - 1 if self.i - self._degree == 0 else self.i - self._degree - 1
        y = 0 if self.j + 1 == n else self.j + 1

        self._gain = 0.0
        self._gain += data.get_cost(self._sequence[self.j], self._sequence[self.i] if self._withTwoOpt else self._sequence[self.i - self._degree])
        self._gain += data.get_cost(self._sequence[self.i - self._degree] if self._withTwoOpt else self._sequence[self.i], self._sequence[y])
        self._gain -= data.get_cost(self._sequence[self.j], self._sequence[y])
        self._gain -= data.get_cost(self._sequence[self.i], self._sequence[self.i + 1])
        self._gain += data.get_cost(self._sequence[x], self._sequence[self.i + 1])
        self._gain -= data.get_cost(self._sequence[x], self._sequence[self.i - self._degree])