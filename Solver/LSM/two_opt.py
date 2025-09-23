from Solver.Moves import Move
from Data.InputData import InputData
from .LocalSearchMoves import LocalSearchMove

class TwoOptMove(LocalSearchMove):
    """2-opt move that reverses the segment between indices i and j inclusive."""

    def set_gain(self, data: InputData) -> None:
        n = len(self._sequence)
        # Full tour reversal (no change) or trivial
        if self.i >= self.j or (self.i == 0 and self.j == n - 1):
            self._gain = 0.0
            return

        # borders management
        x = self.i - 1 if self.i > 0 else n - 1
        y = self.j + 1 if self.j + 1 < n else 0

        self._gain = 0.0
        self._gain -= data.get_cost(self._sequence[x], self._sequence[self.i])
        self._gain -= data.get_cost(self._sequence[self.j], self._sequence[y])
        self._gain += data.get_cost(self._sequence[x], self._sequence[self.j])
        self._gain += data.get_cost(self._sequence[self.i], self._sequence[y])

    def perform(self) -> None:
        if self._gain <= 0:
            Move(self.i, self.j).two_opt(self._sequence)
