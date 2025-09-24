from Solver.Moves import move
from Data.InputData import input_data
from .LocalSearchMoves import LocalSearchMove

class RightShiftMove(LocalSearchMove):
    """Right-shift move that moves the element at index j to index i, shifting others left."""

    def __init__(self, sequence, i, j, degree: int = 1, withTwoOpt: bool = True) -> None:
        super().__init__(sequence, i, j)
        self._degree = degree
        self._withTwoOpt = withTwoOpt

    def set_gain(self, data: input_data) -> None:
        n = len(self._sequence)
        # Trivial case and gain computation
        if self.i == self.j + self._degree or (self.i == 0 and self.j + self._degree + 1 == n):
            self._gain = 0
            return
        # borders management
        x = n - 1 if self.i == 0 else self.i - 1
        y = 0 if self.j + self._degree + 1 == n else self.j + self._degree + 1

        self._gain = 0.0

        self._gain += data.get_cost(self._sequence[self.j] if self._withTwoOpt else self._sequence[self.j + self._degree], self._sequence[self.i])
        self._gain += data.get_cost(self._sequence[x], self._sequence[self.j + self._degree] if self._withTwoOpt else self._sequence[self.j])
        self._gain -= data.get_cost(self._sequence[x], self._sequence[self.i])
        self._gain -= data.get_cost(self._sequence[self.j - 1], self._sequence[self.j])
        self._gain -= data.get_cost(self._sequence[self.j + self._degree], self._sequence[y])
        self._gain += data.get_cost(self._sequence[self.j - 1], self._sequence[y])

    def perform(self) -> None:
        if self._gain < 0:
            for k in range(self._degree + 1):
                m = move(self.i if self._withTwoOpt else self.i + k, self.j + k)
                m.right_shift(self._sequence)