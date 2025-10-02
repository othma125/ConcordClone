from TSPSolver.Moves import move
from TSPData.TSPInstance import TSPInstance
from .LocalSearchMoves import LocalSearchMove


class two_opt_move(LocalSearchMove):
    """2-opt move that reverses the segment between indices i and j inclusive."""

    def set_gain(self, data: TSPInstance) -> None:
        n = len(self._sequence)
        # Full tour reversal (no change) or trivial
        if self.i >= self.j or (self.i == 0 and self.j == n - 1):
            self._gain = 0.0
            return

        # borders management
        x = self.i - 1 if self.i > 0 else n - 1
        y = self.j + 1 if self.j + 1 < n else 0

        self._gain = 0.0
        seq = self._sequence
        self._gain -= data.get_cost(seq[x], seq[self.i])
        self._gain -= data.get_cost(seq[self.j], seq[y])
        self._gain += data.get_cost(seq[x], seq[self.j])
        self._gain += data.get_cost(seq[self.i], seq[y])

    def perform(self) -> None:
        if self._gain < 0:
            move(self.i, self.j).two_opt(self._sequence)
