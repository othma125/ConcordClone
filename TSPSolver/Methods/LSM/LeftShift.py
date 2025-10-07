from TSPSolver.Methods.LSM.Moves import move
from TSPData.TSPInstance import TSPInstance
from .LocalSearchMoves import LocalSearchMove


class left_shift_move(LocalSearchMove):
    """Left-shift move that moves the element at index j to index i, shifting others right."""

    def __init__(self, sequence, i, j, degree: int = 1, withTwoOpt: bool = True) -> None:
        super().__init__(sequence, i, j)
        self._degree = degree
        self._withTwoOpt = withTwoOpt

    def set_gain(self, data: TSPInstance) -> None:
        n = len(self._sequence)
        # Trivial case and gain computation
        if self.i - self._degree == self.j or (self.i - self._degree == 0 and self.j + 1 == n):
            self._gain = 0.0
            return
        # borders management
        x = n - 1 if self.i - self._degree == 0 else self.i - self._degree - 1
        y = 0 if self.j + 1 == n else self.j + 1
        seq = self._sequence
        self._gain = 0.0
        self._gain += data.get_cost(seq[self.j], seq[self.i] if self._withTwoOpt else seq[self.i - self._degree])
        self._gain += data.get_cost(seq[self.i - self._degree] if self._withTwoOpt else seq[self.i], seq[y])
        self._gain -= data.get_cost(seq[self.j], seq[y])
        self._gain -= data.get_cost(seq[self.i], seq[self.i + 1])
        self._gain += data.get_cost(seq[x], seq[self.i + 1])
        self._gain -= data.get_cost(seq[x], seq[self.i - self._degree])

    def perform(self) -> None:
        if self._gain < 0:
            n = len(self._sequence)
            # Base start index for the block being moved leftwards
            for k in range(self._degree + 1):
                src = self.i - k
                dst = self.j if self._withTwoOpt else self.j - k
                # Validate indices
                if src < 0 or dst < 0 or src >= n or dst >= n:
                    continue  # skip invalid
                if src >= dst:  # left_shift expects src < dst
                    continue
                move(src, dst).left_shift(self._sequence)
