import math
import random
import numpy as np

from Data.InputData import InputData
from Solver.LSM.LocalSearchMoves import LocalSearchMove
from Solver.LSM.TwoOpt import TwoOptMove
from Solver.LSM.LeftShift import LeftShiftMove
from Solver.LSM.RightShift import RightShiftMove
from Solver.LSM.Swap import SwapMove
from Solver.Moves import Move  # rename file if needed (current file name is Swap.py)


class Tour:
    """
    Tour representation with local search using existing move classes:
      - TwoOptMove
      - LeftShiftMove / RightShiftMove (act as insertion variants)
      - SwapMove
    Improvement criterion: move.gain <= 0 (consistent with current move implementations).
    """

    def __init__(self, data: InputData, sequence: np.ndarray = None) -> None:
        """
        Initialize a Tour instance.
        """
        n = data.stops_count
        if sequence is None:
            self._sequence = np.random.permutation(n).astype(int)
        else:
            arr = np.fromiter(sequence, dtype=int)
            if len(arr) != n:
                raise ValueError("Provided sequence length does not match data.stops_count")
            self._sequence = arr.copy()

        self._compute_cost(data)
        self.local_search(data)

    # -------------------- Core utilities --------------------

    def _compute_cost(self, data: InputData) -> None:
        """Calculate total cost of the current sequence."""
        self._cost = 0.0
        i = 0
        while i < len(self._sequence) - 1:
            self._cost += data.get_cost(int(self._sequence[i]), int(self._sequence[i + 1]))
            i += 1
        self._cost += data.get_cost(int(self._sequence[i]), int(self._sequence[0]))

    def get_cost(self) -> float:
        return self._cost

    def get_sequence(self) -> np.ndarray:
        return self._sequence

    # -------------------- Local Search --------------------

    def local_search(self, data: InputData) -> None:
        """
        Iteratively apply 2_opt moves to improve the tour cost
        Stops when no stagnation is reached, then stagnation breaker is applied.
        """
        n = len(self._sequence)
        if n < 2:
            return
        probability: float = math.sqrt(n) / n  # ~1/sqrt(n) expected moves per round
        improved: bool = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Evaluate all move types
                lsm: LocalSearchMove = TwoOptMove(self._sequence, i, j)
                if lsm.get_gain(data) < 0:
                    improved = True
                    lsm.perform()
                    self._cost += lsm.gain
        if improved:
            move: Move = Move(0, n - 1)
            random.randint(0, 10)
            move.right_shift(self._sequence)
            self.local_search(data)  # Recursive call until no improvement
        elif random.random() < probability and self.stagnation_breaker(data):
            self.local_search(data)  # Random perturbation to escape local minima

    def stagnation_breaker(self, data: InputData) -> bool:
        n = len(self._sequence)
        for i in range(0, n - 1):
            best_lsm = None
            for j in range(i + 1, n):
                if j > i + 1:
                    lsm = SwapMove(self._sequence, i, j)
                    if lsm.get_gain(data) < 0 and (best_lsm is None or lsm.get_gain(data) < best_lsm.gain):
                        best_lsm = lsm
                for degree in range(1 if j == i + 1 else 0, 3):
                    if j + degree >= n:
                        break
                    lsm1 = LeftShiftMove(self._sequence, i, j, degree)
                    if lsm1.get_gain(data) < 0 and (best_lsm is None or lsm1.get_gain(data) < best_lsm.gain):
                        best_lsm = lsm1
                    if degree == 0:
                        continue
                    lsm2 = LeftShiftMove(self._sequence, i, j, degree, False)
                    if lsm2.get_gain(data) < 0 and (best_lsm is None or lsm2.get_gain(data) < best_lsm.gain):
                        best_lsm = lsm2

                for degree in range(1 if j == i + 1 else 0, 3):
                    if i - degree < 0:
                        break
                    lsm1 = RightShiftMove(self._sequence, i, j, degree)
                    if lsm1.get_gain(data) < 0 and (best_lsm is None or lsm1.get_gain(data) < best_lsm.gain):
                        best_lsm = lsm1
                    if degree == 0:
                        continue
                    lsm2 = RightShiftMove(self._sequence, i, j, degree, False)
                    if lsm2.get_gain(data) < 0 and (best_lsm is None or lsm2.get_gain(data) < best_lsm.gain):
                        best_lsm = lsm2
            if best_lsm is not None:
                best_lsm.perform()
                return True
        return False

    def __str__(self) -> str:
        return f"cost = {self._cost:.2f} \nSequence = {self.pretty()}"

    def pretty(self) -> str:
        return " -> ".join(str(int(x) + 1) for x in self._sequence) + f" -> {1 + int(self._sequence[0])}"


# Simple manual test (run directly)
if __name__ == "__main__":
    from Data.InputData import InputData
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tsp = os.path.join(root, "ALL_tsp", "bier127.tsp")
    data = InputData(tsp)
    tour = Tour(data)
    print(tour)
