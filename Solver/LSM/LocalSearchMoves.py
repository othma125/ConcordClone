from Data.InputData import InputData
import numpy as np

class LocalSearchMove:
    def __init__(self, sequence: np.ndarray, i: int, j: int) -> None:
        self._sequence: np.ndarray = sequence
        self._i: int = i
        self._j: int = j
        self._gain: float = 0.0  # ensure defined

    @property
    def i(self) -> int:
        return self._i

    @property
    def j(self) -> int:
        return self._j

    @property
    def gain(self) -> float:
        return self._gain
    
    def gain(self, data: InputData) -> float:
        """Compute and return gain."""
        self.set_gain(data)
        return self._gain

    def set_gain(self, data: InputData) -> None:
        """Override in subclasses."""
        pass

    def perform(self) -> None:
        """Override in subclasses."""
        pass