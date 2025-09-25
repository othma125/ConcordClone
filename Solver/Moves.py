import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class move:
    index1: int
    index2: int

    def right_shift(self, arr: np.ndarray) -> None:
        """
        Insert element at index2 into position index1 shifting the window [index1, index2-1] right.
        No-op if index1 >= index2.
        """
        if self.index1 < self.index2:
            aux = arr[self.index2]
            arr[self.index1 + 1: self.index2 + 1] = arr[self.index1: self.index2]
            arr[self.index1] = aux

    def left_shift(self, arr: np.ndarray) -> None:
        """
        Move element at index1 to index2 shifting window [index1+1, index2] left.
        No-op if index1 >= index2.
        """
        if self.index1 < self.index2:
            aux = arr[self.index1]
            arr[self.index1: self.index2] = arr[self.index1 + 1: self.index2 + 1]
            arr[self.index2] = aux

    def swap(self, arr: np.ndarray) -> None:
        """exchange elements at index1 and index2."""
        arr[self.index1], arr[self.index2] = arr[self.index2], arr[self.index1]

    def two_opt(self, arr: np.ndarray) -> None:
        """
        Reverse subsequence between index1 and index2 inclusive (classic 2-opt edge reversal).
        No-op if index1 >= index2.
        """
        if self.index1 < self.index2:
            arr[self.index1: self.index2 + 1] = arr[self.index1: self.index2 + 1][::-1]


# Optional simple tests
if __name__ == "__main__":
    a = np.array([0, 1, 2, 3, 4, 5, 6])
    move(1, 5).right_shift(a)  # move 5 to index 1
    assert np.all(a == np.array([0, 5, 1, 2, 3, 4, 6]))
    move(1, 5).left_shift(a)  # undo previous move
    assert np.all(a == np.array([0, 1, 2, 3, 4, 5, 6]))
    move(2, 4).swap(a)
    assert np.all(a == np.array([0, 1, 4, 3, 2, 5, 6]))
    move(2, 5).two_opt(a)  # reverse indices 2..5
    # segment [4,3,2,5] becomes [5,2,3,4]
    assert np.all(a == np.array([0, 1, 5, 2, 3, 4, 6]))
    print("OK")
