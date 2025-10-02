from TSPData.TSPInstance import TSPInstance
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
    
    def __lt__(self, other: 'LocalSearchMove') -> bool:
        return self._gain < other._gain

    def get_gain(self, data: TSPInstance) -> float:
        """Compute and return gain."""
        self.set_gain(data)
        return self._gain

    def set_gain(self, data: TSPInstance) -> None:
        """Override in subclasses."""
        pass

    def perform(self) -> None:
        """Override in subclasses."""
        pass