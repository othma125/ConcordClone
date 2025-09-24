from Solver.Moves import move
from Data.InputData import input_data
from .LocalSearchMoves import LocalSearchMove


class swap_move(LocalSearchMove):
    """Swap move that exchanges the elements at indices i and j."""

    def set_gain(self, data: input_data) -> None:
        n = len(self._sequence)
        # Trivial case
        if self.i == self.j or (self.i == 0 and self.j + 1 == n):
            self._gain = 0.0
            return
        # borders management
        x = n - 1 if self.i == 0 else self.i - 1
        y = 0 if self.j + 1 == n else self.j + 1
        self._gain = 0.0
        seq = self._sequence
        self._gain += data.get_cost(seq[x], seq[self.j]) - data.get_cost(seq[x], seq[self.i])
        self._gain += data.get_cost(seq[self.j - 1], seq[self.i]) - data.get_cost(seq[self.j - 1], seq[self.j])
        self._gain += data.get_cost(seq[self.j], seq[self.i + 1]) - data.get_cost(seq[self.i], seq[self.i + 1])
        self._gain += data.get_cost(seq[self.i], seq[y]) - data.get_cost(seq[self.j], seq[y])

    def perform(self):
        if self._gain < 0:
            move(self.i, self.j).swap(self._sequence)
