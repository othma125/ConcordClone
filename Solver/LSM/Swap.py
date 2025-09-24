from Solver.Moves import move
from Data.InputData import input_data
from .LocalSearchMoves import LocalSearchMove

class SwapMove(LocalSearchMove):
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
        self._gain += data.get_cost(self._sequence[x], self._sequence[self.j]) - data.get_cost(self._sequence[x], self._sequence[self.i])
        self._gain += data.get_cost(self._sequence[self.j - 1], self._sequence[self.i]) - data.get_cost(self._sequence[self.j - 1], self._sequence[self.j])
        self._gain += data.get_cost(self._sequence[self.j], self._sequence[self.i + 1]) - data.get_cost(self._sequence[self.i], self._sequence[self.i + 1])
        self._gain += data.get_cost(self._sequence[self.i], self._sequence[y]) - data.get_cost(self._sequence[self.j], self._sequence[y])

    def perform(self):
        if self._gain < 0:
            move(self.i, self.j).swap(self._sequence)