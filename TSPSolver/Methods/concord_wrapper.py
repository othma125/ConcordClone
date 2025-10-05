from TSPSolver.TSPTour import TSPTour
from TSPSolver.Methods.TSPSolver import TSPSolver
from time import time
import numpy as np
from pathlib import Path


class ConcordeWrapperSolver(TSPSolver):
    def __init__(self, file_name: str):
        super().__init__(file_name)

    def solve(self, max_time: float = float("inf")) -> TSPTour:
        """Run Concorde via the pyconcorde wrapper.

        This function imports the `pyconcorde` package from the Python path.
        If you cloned the `pyconcorde` repository locally (for example into
        a sibling folder or into the project), Python will import that local
        package automatically when running from the repository root.

        Recommended ways to make the package available:
        - Clone locally and install in editable/development mode (recommended
          when you want to inspect or modify pyconcorde):

            git clone https://github.com/jvkersch/pyconcorde.git /path/to/pyconcorde
            pip install -e /path/to/pyconcorde

        - Or install directly from GitHub (non-editable):

            pip install "pyconcorde @ git+https://github.com/jvkersch/pyconcorde"

        Note: building pyconcorde on native Windows may fail because the
        package tries to build Concorde/QSopt; using WSL/Linux or a container
        is the most reliable approach if you need the full Concorde binary.
        """
        if max_time <= 0:
            raise ValueError("max_time must be positive or infinity")
        try:
            from pyconcorde.concorde.tsp import TSPSolver as ConcordeSolver
            from pyconcorde.concorde.util import write_tsp_file
        except Exception as e:
            msg = str(e)
            if isinstance(e, ModuleNotFoundError) or "concorde" in msg:
                print(
                    "Warning: pyconcorde package was found but the compiled Concorde"
                    " extension is missing (this is common on Windows). Falling back to"
                    " 'pyvrp_hgs' solver. To use Concorde, install/build Concorde (or"
                    " run in WSL/Linux) and install pyconcorde. Original error:\n",
                    e,
                )
                from TSPSolver.Methods.pyvrp_hgs import PyvrpHGSSolver
                return PyvrpHGSSolver(self._data).solve()
            raise ImportError(
                "pyconcorde could not be imported. If you cloned the project locally,"
                " make sure it is on the Python path or installed in editable mode.\n"
                f"Original import error: {e}"
            )

        start_time = time()
        n = self._data.stops_count
        if n > 10000:
            raise ValueError("Concorde wrapper supports up to 10,000 stops due to memory constraints.")
        print(f"File = {self._file_name.split('\\')[-1]}")
        print(f"Stops Count = {n}")
        print("Solution approach = Concord TSP (pyconcorde)")

        repo_root = Path(__file__).resolve().parent.parent
        tsp_dir = repo_root / "DefaultInstances" / "TSPLIB"
        selected_file = tsp_dir / self._file_name
        if not selected_file.is_file():
            raise FileNotFoundError(f"TSP file not found: {selected_file}")

        solver_c = ConcordeSolver.from_tspfile(str(selected_file))
        time_bound = -1 if max_time == float('inf') else float(max_time)
        comp = solver_c.solve(time_bound=time_bound, verbose=False, random_seed=0)

        cycle = np.array(comp.tour, dtype=int) - 1
        if len(cycle) == n + 1 and cycle[-1] == -1:
            cycle = cycle[:-1]
        if len(cycle) != n:
            raise RuntimeError(f"Concorde returned cycle of length {len(cycle)} for n={n}")

        new_tour = TSPTour(self._data, cycle)
        new_tour.set_reach_time(time() - start_time)
        new_tour.set_solution_methode("Concorde via pyconcorde")
        print(f"New best cost = {new_tour.cost:.2f} at {int(new_tour.reach_time * 1000)} ms")
        return new_tour
