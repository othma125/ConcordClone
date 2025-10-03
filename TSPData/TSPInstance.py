import math
import os
import re
import threading
from typing import Dict, List, Optional, TextIO
import tracemalloc
import numpy as np
from .Location import Location
from .Edge import Edge


class TSPInstance:
    DEFAULT_MATRIX_THRESHOLD = 150
    MATRIX_ENV = "TSP_MATRIX_MAX"

    def __init__(self, path: str, max_dimension: int = float('inf')):
        self._file_name = path if isinstance(path, str) else getattr(path, "name", "UNKNOWN")
        self._header: Dict[str, str] = {}
        self._coordinates: List[Location] = []  # type: ignore
        self.stops_count: int = 0
        self.use_matrix: bool = False
        self.explicit_weights: bool = False
        self._cost_matrix: Optional[np.ndarray] = None  # CHANGED
        self._cost_map: Optional[Dict[Edge, float]] = None  # type: ignore
        self._closed = False
        self._lock = threading.RLock()

        with (open(path, "r", encoding="utf-8", errors="ignore") if isinstance(path, (str, os.PathLike)) else path) as fh:
            early_exit = self._parse_tsplib(fh, max_dimension)
            if early_exit:
                return
            if self.use_matrix and self.stops_count > 0:
                self._allocate_matrix()

        if not self.use_matrix and self._cost_map is None:
            self._cost_map = {}
    @property
    def matrix(self) -> Optional[np.ndarray]:
        if not self.use_matrix:
            return None
        return self._cost_matrix
    
    @property
    def coordinates(self) -> List[Location]:
        if hasattr(self, '_coordinates'):
            return self._coordinates
        raise ValueError("no attribute defined with this name")

    def _parse_tsplib(self, fh: TextIO, max_dimension: int) -> bool:
        section = None

        # Read header until a section starts
        while True:
            line = fh.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("NODE_COORD_SECTION"):
                section = "NODE_COORD_SECTION"
                break
            elif upper.startswith("EDGE_WEIGHT_SECTION"):
                section = "EDGE_WEIGHT_SECTION"
                break
            elif upper.startswith("EOF"):
                break
            else:
                if ":" in line:
                    k, v = line.split(":", 1)
                    self._header[k.strip().upper()] = v.strip()
                else:
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        self._header[parts[0].strip().upper()] = parts[1].strip()

        if "DIMENSION" in self._header:
            dim_str = re.sub(r"[^0-9]", "", self._header["DIMENSION"])
            if dim_str:
                self.stops_count = int(dim_str)
            if self.stops_count > max_dimension:
                return True

        if self.stops_count > 0:
            self.use_matrix = self._decide_matrix_strategy()
        else:
            raise ValueError("DIMENSION is required in TSPLIB file")

        edge_weight_type = self._header.get("EDGE_WEIGHT_TYPE", "").upper()
        edge_weight_format = self._header.get("EDGE_WEIGHT_FORMAT", "").upper()

        if edge_weight_type == "EXPLICIT" or section == "EDGE_WEIGHT_SECTION":
            self.explicit_weights = True
            self._read_explicit_weights(fh, edge_weight_format)
        else:
            if section is None or section != "NODE_COORD_SECTION":
                # scan forward
                for line in fh:
                    if line.strip().upper().startswith("NODE_COORD_SECTION"):
                        break
            self._read_node_coords(fh)

        return False

    # Decide between dense matrix or map
    def _decide_matrix_strategy(self) -> bool:
        n = self.stops_count
        if n <= self.DEFAULT_MATRIX_THRESHOLD:
            return True
        # ENV override
        try:
            override = int(os.environ.get(self.MATRIX_ENV, "-1"))
            if override > 0 and n <= override:
                return self._has_heap_for_matrix(0.5)
        except ValueError:
            pass
        return self._has_heap_for_matrix(0.35)

    def _estimate_full_matrix_bytes(self) -> int:
        n = self.stops_count
        return 8 * n * n + 16 * n + 64

    def _has_heap_for_matrix(self, allowance_ratio: float) -> bool:
        # Best-effort heuristic (no psutil required)
        try:
            tracemalloc.start()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            remaining = max(512 * 1024 * 1024 - current, 1)  # assume at least 512MB if cannot detect
        except Exception:
            remaining = 512 * 1024 * 1024
        needed = self._estimate_full_matrix_bytes()
        return needed < remaining * allowance_ratio

    # --------------- Coordinate & Explicit Parsing --------------- #

    def _read_node_coords(self, fh: TextIO):
        self._coordinates.clear()
        for raw in fh:
            t = raw.strip()
            if not t:
                continue
            u = t.upper()
            if u.startswith("EOF") or u.startswith("EDGE_WEIGHT_SECTION") or u.startswith(
                    "DISPLAY_DATA_SECTION") or u.startswith("TOUR_SECTION"):
                break
            parts = t.split()
            if len(parts) < 2:
                continue
            start = 0
            if len(parts) >= 3 and self._is_number(parts[0]):
                start = 1
            if len(parts) - start < 2:
                continue
            x = self._parse_float(parts[start])
            y = self._parse_float(parts[start + 1])
            if x is not None and y is not None:
                self._coordinates.append(Location(x, y))
                if 0 < self.stops_count <= len(self._coordinates):
                    break
        if self.stops_count and len(self._coordinates) < self.stops_count:
            self.stops_count = len(self._coordinates)

    def _allocate_matrix(self):
        if self._cost_matrix is not None:
            return
        # Use np.nan for not yet computed (except diagonal implicitly zero)
        self._cost_matrix = np.full((self.stops_count, self.stops_count), np.nan, dtype=float)

    def get_cost(self, x: int, y: int) -> float:
        self._ensure_open()
        if x < 0 or y < 0 or x >= self.stops_count or y >= self.stops_count:
            raise IndexError("Invalid node index")
        if x == y:
            return 0.0
        if self.use_matrix:
            existing = self._cost_matrix[x, y]
            if self.explicit_weights:
                if np.isnan(existing):
                    raise RuntimeError(f"Missing explicit matrix value for {Edge}")
                return existing
            if not np.isnan(existing) and existing > 0:
                return existing
            return self._compute_and_store(x, y)
        else:
            val = self._cost_map.get(Edge(x, y))
            if val is not None:
                return val
            if self.explicit_weights:
                raise RuntimeError(f"Missing explicit edge {Edge}")
            return self._compute_and_store(x, y)

    def _compute_and_store(self, x: int, y: int) -> float:
        with self._lock:
            loc1 = self._coordinates[x]
            loc2 = self._coordinates[y]
            edge_weight_type = self._header.get("EDGE_WEIGHT_TYPE", "").upper()
            if edge_weight_type == "CEIL_2D":
                cost = math.ceil(loc1.get_euclidean(loc2))
            elif edge_weight_type.startswith("EUC"):
                cost = round(loc1.get_euclidean(loc2))
            elif edge_weight_type == "GEO":
                cost = loc1.to_geo().get_geo_great_circle_distance(loc2.to_geo())
            elif edge_weight_type == "ATT":
                cost = loc1.get_pseudo_euclidean_distance(loc2)
            else:
                cost = loc1.get_euclidean(loc2)

            if self.use_matrix:
                self._cost_matrix[x, y] = cost
                self._cost_matrix[y, x] = cost
            else:
                self._cost_map[Edge(x, y)] = cost
                self._cost_map[Edge(y, x)] = cost
            return cost

    def _read_explicit_weights(self, fh: TextIO, fmt: str):
        if self.stops_count <= 0:
            raise ValueError("DIMENSION must be specified for EXPLICIT instances")
        fmt = (fmt or "FULL_MATRIX").upper()
        if self.use_matrix:
            self._allocate_matrix()
        else:
            if self._cost_map is None:
                self._cost_map = {}
        nums: List[float] = []
        for raw in fh:
            t = raw.strip()
            if not t:
                continue
            u = t.upper()
            if u.startswith("EOF") or u.startswith("DISPLAY_DATA_SECTION") or u.startswith(
                    "NODE_COORD_SECTION") or u.startswith("TOUR_SECTION"):
                break
            for tok in t.split():
                v = self._parse_float(tok)
                if v is not None:
                    nums.append(v)

        expected = self._expected_explicit_count(fmt)
        n = self.stops_count
        if len(nums) != expected:
            raise ValueError(f"Mismatch weight count. Expected {expected} ({fmt}), found {len(nums)}")

        idx = 0
        if self.use_matrix:
            if self._cost_matrix is None:
                raise RuntimeError("Matrix not allocated")
            if fmt == "FULL_MATRIX":
                for i in range(n):
                    for j in range(n):
                        self._cost_matrix[i, j] = nums[idx]
                        idx += 1
            elif fmt == "UPPER_ROW":
                for i in range(n):
                    for j in range(i + 1, n):
                        w = nums[idx]
                        idx += 1
                        self._cost_matrix[i, j] = w
                        self._cost_matrix[j, i] = w
            elif fmt == "LOWER_ROW":
                for i in range(n):
                    for j in range(i):
                        w = nums[idx]
                        idx += 1
                        self._cost_matrix[i, j] = w
                        self._cost_matrix[j, i] = w
            elif fmt == "UPPER_DIAG_ROW":
                for i in range(n):
                    for j in range(i, n):
                        w = nums[idx]
                        idx += 1
                        self._cost_matrix[i, j] = w
                        self._cost_matrix[j, i] = w
            elif fmt == "LOWER_DIAG_ROW":
                for i in range(n):
                    for j in range(i + 1):
                        w = nums[idx]
                        idx += 1
                        self._cost_matrix[i, j] = w
                        self._cost_matrix[j, i] = w
            else:
                raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {fmt}")
        else:
            if fmt == "FULL_MATRIX":
                for i in range(n):
                    for j in range(n):
                        self._cost_map[Edge(i, j)] = nums[idx]
                        idx += 1
            elif fmt == "UPPER_ROW":
                for i in range(n):
                    for j in range(i + 1, n):
                        w = nums[idx]
                        idx += 1
                        self._cost_map[Edge(i, j)] = w
                        self._cost_map[Edge(j, i)] = w
            elif fmt == "LOWER_ROW":
                for i in range(n):
                    for j in range(i):
                        w = nums[idx]
                        idx += 1
                        self._cost_map[Edge(i, j)] = w
                        self._cost_map[Edge(j, i)] = w
            elif fmt == "UPPER_DIAG_ROW":
                for i in range(n):
                    for j in range(i, n):
                        w = nums[idx]
                        idx += 1
                        self._cost_map[Edge(i, j)] = w
                        self._cost_map[Edge(j, i)] = w
            elif fmt == "LOWER_DIAG_ROW":
                for i in range(n):
                    for j in range(i + 1):
                        w = nums[idx]
                        idx += 1
                        self._cost_map[Edge(i, j)] = w
                        self._cost_map[Edge(j, i)] = w
            else:
                raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {fmt}")
            for i in range(n):
                self._cost_map.setdefault(Edge(i, i), 0.0)

    # --------------- Utilities --------------- #

    @staticmethod
    def _parse_float(s: str) -> Optional[float]:
        try:
            return float(s)
        except ValueError:
            return None

    @staticmethod
    def _is_number(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    def __del__(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._cost_map is not None:
            self._cost_map.clear()
        if self._cost_matrix is not None:
            self._cost_matrix = None  # np.ndarray, just dereference
        self._coordinates.clear()
        self._header.clear()

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError(f"InputData instance already closed: {self._file_name}")
