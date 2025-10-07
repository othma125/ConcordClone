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
