import math
import numpy as np

from Data.InputData import input_data
from Solver.LSM.LocalSearchMoves import LocalSearchMove
from Solver.LSM.TwoOpt import two_opt_move
from Solver.LSM.LeftShift import left_shift_move
from Solver.LSM.RightShift import right_shift_move
from Solver.LSM.Swap import swap_move
from Solver.Moves import move  # rename file if needed (current file name is Swap.py)


class tour:
    """
    Tour representation with local search using existing move classes:
      - TwoOptMove
      - LeftShiftMove / RightShiftMove (act as insertion variants)
      - SwapMove
    Improvement criterion: move.gain <= 0 (consistent with current move implementations).
    """

    def __init__(self, data: input_data, sequence: np.ndarray = None, improve: bool = False) -> None:
        """
        Initialize a Tour instance.
        """
        n = data.stops_count
        if sequence is None:
            improve = True
            self._sequence = np.random.permutation(n).astype(int)
        else:
            if len(sequence) != n:
                raise ValueError("Provided sequence length does not match data.stops_count")
            self._sequence = np.fromiter(sequence, dtype=int).copy()
        self._compute_cost(data)
        if improve:
            self._local_search(data)

    # -------------------- Core utilities --------------------

    def _compute_cost(self, data: input_data) -> None:
        """Calculate total cost of the current sequence."""
        self._cost = 0.0
        i = 0
        n = len(self._sequence)
        while i < n - 1:
            self._cost += data.get_cost(int(self._sequence[i]), int(self._sequence[i + 1]))
            i += 1
        self._cost += data.get_cost(int(self._sequence[i]), int(self._sequence[0]))
    
    @property
    def cost(self) -> float:
        return self._cost

    @property
    def sequence(self) -> np.ndarray:
        return self._sequence

    # -------------------- Local Search --------------------

    def _local_search(self, data: input_data) -> None:
        """
        Iteratively apply 2_opt moves to improve the tour cost
        Stops when no stagnation is reached, then stagnation breaker is applied with some probability
        """
        n = len(self._sequence)
        if n < 2:
            return
        probability: float = math.sqrt(n) / n  # ~1/sqrt(n) expected moves per round
        improved: bool = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Evaluate all move types
                lsm: LocalSearchMove = two_opt_move(self._sequence, i, j)
                if lsm.get_gain(data) < 0:
                    improved = True
                    lsm.perform()
                    self._cost += lsm.gain
                    break
        if improved:
            m = move(0, n - 1)
            iterations = np.random.randint(0, 10)
            for _ in range(iterations):
                m.right_shift(self._sequence)
            self._local_search(data)  # Recursive call until no improvement
        elif np.random.random() < probability and self._stagnation_breaker(data):
            self._local_search(data)  # Random perturbation to escape local minima

    def _stagnation_breaker(self, data: input_data) -> bool:
        """ Try to find any improving move to escape stagnation."""
        n = len(self._sequence)
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                if j > i + 1:
                    lsm = swap_move(self._sequence, i, j)
                    if lsm.get_gain(data) < 0:
                        lsm.perform()
                        return True
                for degree in range(1 if j == i + 1 else 0, 3):
                    if i - degree < 0:
                        break
                    lsm1 = left_shift_move(self._sequence, i, j, degree)
                    if lsm1.get_gain(data) < 0:
                        lsm1.perform()
                        return True
                    if degree == 0:
                        continue
                    lsm2 = left_shift_move(self._sequence, i, j, degree, False)
                    if lsm2.get_gain(data) < 0:
                        lsm2.perform()
                        return True

                for degree in range(1 if j == i + 1 else 0, 3):
                    if j + degree >= n:
                        break
                    lsm1 = right_shift_move(self._sequence, i, j, degree)
                    if lsm1.get_gain(data) < 0:
                        lsm1.perform()
                        return True
                    if degree == 0:
                        continue
                    lsm2 = right_shift_move(self._sequence, i, j, degree, False)
                    if lsm2.get_gain(data) < 0:
                        lsm2.perform()
                        return True
        return False

    def perturbation(self, data: input_data) -> 'tour':
        """ two bridge-like perturbation to escape local minima """
        n = len(self._sequence)
        if n < 8 or np.random.random() < 0.3:
            return tour(data)  # Random new tour
        quarter = max(1, n // 4)

        i = 1 + np.random.randint(0, quarter - 1)
        j = i + 1 + np.random.randint(0, quarter - 1)
        k = j + 1 + np.random.randint(0, quarter - 1)

        k = min(k, n)
        j = min(j, k - 1)

        new_seq = np.concatenate([
            self._sequence[:i],
            self._sequence[j:k],
            self._sequence[i:j],
            self._sequence[k:]
        ])
        return tour(data, new_seq, True)

    def set_reach_time(self, time: float) -> None:
        setattr(self, '_reach_time', time)

    @property
    def reach_time(self) -> float:
        if hasattr(self, '_reach_time'):
            return self._reach_time
        raise ValueError("no attribute defined with this name")
    
    def __str__(self) -> str:
        return f"cost = {self._cost:.2f} \nreach time = {int(self.reach_time * 1000)} ms\nSequence = {self._pretty()}"

    def _pretty(self) -> str:
        return " -> ".join(str(int(x) + 1) for x in self._sequence) + f" -> {1 + int(self._sequence[0])}"
    
    def __lt__(self, other: 'tour') -> bool:
        return self._cost < other._cost
    
    def __del__(self) -> None:
        del self._sequence
