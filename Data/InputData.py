import math
import os
import re
import threading
from typing import Dict, Tuple, List, Optional, TextIO
import tracemalloc
from .Edge import Edge
from .Location import Location


class InputData:
    DEFAULT_MATRIX_THRESHOLD = 150  # Always matrix below or equal
    MATRIX_ENV = "TSP_MATRIX_MAX"  # Override like export TSP_MATRIX_MAX=500

    def __init__(self, path: str, max_dimension: int = float('inf')):
        self.file_name = path if isinstance(path, str) else getattr(path, "name", "UNKNOWN")
        self.header: Dict[str, str] = {}
        self.coordinates: List[Location] = []
        self.stops_count: int = 0
        self.use_matrix: bool = False
        self.explicit_weights: bool = False
        self._cost_matrix: Optional[List[List[Optional[float]]]] = None
        self._cost_map: Optional[Dict[Tuple[int, int], float]] = None
        self._closed = False
        self._lock = threading.RLock()

        need_allocate_after_parse = False
        with (open(path, "r", encoding="utf-8", errors="ignore") if isinstance(path, (str, os.PathLike)) else path) as fh:
            early_exit = self._parse_tsplib(fh, max_dimension)
            if early_exit:
                return
            if self.use_matrix and self.stops_count > 0:
                self._allocate_matrix()
                need_allocate_after_parse = True
            elif not self.use_matrix and self._cost_map is None:
                self._cost_map = {}

        # For coordinate-based + matrix strategy we lazily fill; matrix already allocated
        if not self.use_matrix and self._cost_map is None:
            self._cost_map = {}

    # --------------- Parsing --------------- #

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
                    self.header[k.strip().upper()] = v.strip()
                else:
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        self.header[parts[0].strip().upper()] = parts[1].strip()

        if "DIMENSION" in self.header:
            dim_str = re.sub(r"[^0-9]", "", self.header["DIMENSION"])
            if dim_str:
                self.stops_count = int(dim_str)
            if self.stops_count > max_dimension:
                return True

        if self.stops_count > 0:
            self.use_matrix = self._decide_matrix_strategy()

        edge_weight_type = self.header.get("EDGE_WEIGHT_TYPE", "").upper()
        edge_weight_format = self.header.get("EDGE_WEIGHT_FORMAT", "").upper()

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

    def _allocate_matrix(self):
        if self._cost_matrix is not None:
            return
        # Use None sentinel for not yet computed (except diagonal implicitly zero)
        self._cost_matrix = [[None for _ in range(self.stops_count)] for _ in range(self.stops_count)]

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
        self.coordinates.clear()
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
                self.coordinates.append(Location(x, y))
                if self.stops_count > 0 and len(self.coordinates) >= self.stops_count:
                    break
        if self.stops_count and len(self.coordinates) < self.stops_count:
            self.stops_count = len(self.coordinates)

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
            if len(nums) == n * (n + 1) // 2:
                fmt = "LOWER_DIAG_ROW" if "LOWER" in fmt else "UPPER_DIAG_ROW"
                expected = len(nums)
            elif len(nums) == n * (n - 1) // 2:
                fmt = "LOWER_ROW" if "LOWER" in fmt else "UPPER_ROW"
                expected = len(nums)
            elif len(nums) == n * n:
                fmt = "FULL_MATRIX"
                expected = len(nums)

        if len(nums) != expected:
            raise ValueError(f"Mismatch weight count. Expected {expected} ({fmt}), found {len(nums)}")

        idx = 0
        if self.use_matrix:
            if self._cost_matrix is None:
                raise RuntimeError("Matrix not allocated")
            if fmt == "FULL_MATRIX":
                for i in range(n):
                    for j in range(n):
                        self._cost_matrix[i][j] = nums[idx]
                        idx += 1
            elif fmt == "UPPER_ROW":
                for i in range(n):
                    for j in range(i + 1, n):
                        w = nums[idx]
                        idx += 1
                        self._cost_matrix[i][j] = w
                        self._cost_matrix[j][i] = w
            elif fmt == "LOWER_ROW":
                for i in range(n):
                    for j in range(i):
                        w = nums[idx]
                        idx += 1
                        self._cost_matrix[i][j] = w
                        self._cost_matrix[j][i] = w
            elif fmt == "UPPER_DIAG_ROW":
                for i in range(n):
                    for j in range(i, n):
                        w = nums[idx]
                        idx += 1
                        self._cost_matrix[i][j] = w
                        self._cost_matrix[j][i] = w
            elif fmt == "LOWER_DIAG_ROW":
                for i in range(n):
                    for j in range(i + 1):
                        w = nums[idx]
                        idx += 1
                        self._cost_matrix[i][j] = w
                        self._cost_matrix[j][i] = w
            else:
                raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {fmt}")
        else:
            assert self._cost_map is not None
            if fmt == "FULL_MATRIX":
                for i in range(n):
                    for j in range(n):
                        self._cost_map[(i, j)] = nums[idx]
                        idx += 1
            elif fmt == "UPPER_ROW":
                for i in range(n):
                    for j in range(i + 1, n):
                        w = nums[idx]
                        idx += 1
                        self._cost_map[(i, j)] = w
                        self._cost_map[(j, i)] = w
            elif fmt == "LOWER_ROW":
                for i in range(n):
                    for j in range(i):
                        w = nums[idx]
                        idx += 1
                        self._cost_map[(i, j)] = w
                        self._cost_map[(j, i)] = w
            elif fmt == "UPPER_DIAG_ROW":
                for i in range(n):
                    for j in range(i, n):
                        w = nums[idx]
                        idx += 1
                        self._cost_map[(i, j)] = w
                        self._cost_map[(j, i)] = w
            elif fmt == "LOWER_DIAG_ROW":
                for i in range(n):
                    for j in range(i + 1):
                        w = nums[idx]
                        idx += 1
                        self._cost_map[(i, j)] = w
                        self._cost_map[(j, i)] = w
            else:
                raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {fmt}")
            for i in range(n):
                self._cost_map.setdefault((i, i), 0.0)

    def _expected_explicit_count(self, fmt: str) -> int:
        n = self.stops_count
        if fmt == "FULL_MATRIX":
            return n * n
        if fmt in ("UPPER_ROW", "LOWER_ROW"):
            return n * (n - 1) // 2
        if fmt in ("UPPER_DIAG_ROW", "LOWER_DIAG_ROW"):
            return n * (n + 1) // 2
        raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT: {fmt}")

    # --------------- Cost Retrieval --------------- #

    def get_cost(self, edge: Edge) -> float:
        if type(edge) is not Edge:
            raise TypeError("Expected Edge instance")
        return self.get_cost(edge.X, edge.Y)

    def get_cost(self, x: int, y: int) -> float:
        self._ensure_open()
        if x < 0 or y < 0 or x >= self.stops_count or y >= self.stops_count:
            raise IndexError("Invalid node index")
        if x == y:
            return 0.0
        if self.use_matrix:
            m = self._cost_matrix
            assert m is not None
            existing = m[x][y]
            if self.explicit_weights:
                if existing is None:
                    raise RuntimeError(f"Missing explicit matrix value for {edge}")
                return existing
            if existing is not None and existing > 0:
                return existing
            return self._compute_and_store(x, y)
        else:
            cm = self._cost_map
            assert cm is not None
            val = cm.get((x, y))
            if val is not None:
                return val
            if self.explicit_weights:
                raise RuntimeError(f"Missing explicit edge {edge}")
            return self._compute_and_store(x, y)

    def _compute_and_store(self, x: int, y: int) -> float:
        with self._lock:
            loc1 = self.coordinates[x]
            loc2 = self.coordinates[y]
            edge_weight_type = self.header.get("EDGE_WEIGHT_TYPE", "").upper()
            if edge_weight_type == "CEIL_2D":
                cost = math.ceil(loc1.get_euclidean(loc2))
            elif edge_weight_type.startswith("EUC"):
                cost = int(math.ceil(loc1.get_euclidean(loc2)))
            elif edge_weight_type == "GEO":
                cost = loc1.to_geo().get_geo_great_circle_distance(loc2.to_geo())
            elif edge_weight_type == "ATT":
                cost = loc1.get_pseudo_euclidean_distance(loc2)
            else:
                cost = int(loc1.get_euclidean(loc2) + 0.5)

            if self.use_matrix:
                self._cost_matrix[x][y] = cost
                self._cost_matrix[y][x] = cost
            else:
                self._cost_map[(x, y)] = cost
                self._cost_map[(y, x)] = cost
            return cost

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

    # --------------- Lifecycle --------------- #

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._cost_map is not None:
            self._cost_map.clear()
        if self._cost_matrix is not None:
            self._cost_matrix = None
        self.coordinates.clear()
        self.header.clear()

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError(f"InputData instance already closed: {self.file_name}")

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __str__(self):
        return f"InputData(file='{self.file_name}', stops={self.stops_count})"

    # --------------- Comparable-like --------------- #

    def __lt__(self, other: "InputData"):
        return self.stops_count < other.stops_count

    def __repr__(self):
        strat = "MATRIX" if self.use_matrix else "MAP"
        return f"InputData(file='{self.file_name}', stops={self.stops_count}, strategy={strat}, explicit={self.explicit_weights})"
