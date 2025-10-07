"""
TSPSolver: Lightweight orchestrator for solving Traveling Salesman Problem (TSP) instances.
Classes:
    TSPSolver:
        Orchestrates the loading, solving, and visualization of TSP instances.
        Delegates actual solving to methods registered in TSPSolver.Methods.registry.
Attributes:
    MAX_STOPS (int): Maximum number of stops to visualize clearly.
Methods:
    __init__(self, data: TSPInstance):
        Initializes the TSPSolver with a TSPInstance.
    Solve(self, **kwargs) -> TSPTour:
        Dispatches to the requested TSP solver method from the registry.
        Args:
            method (str): Name of the solver (registry key); defaults to 'christofides'.
            max_time (float): Optional maximum time in seconds for the solver (default: infinity).
        Returns:
            TSPTour: The resulting tour from the solver.
    __del__(self) -> None:
        Destructor to clean up the TSPInstance data.
    Visualisation(self, tour: TSPTour) -> None:
        Draws the TSP tour using networkX and matplotlib if coordinates are available and the number of stops is within MAX_STOPS.
        Args:
            tour (TSPTour): The tour to visualize.
        Notes:
            - Skips drawing if the tour is invalid or the number of stops exceeds MAX_STOPS.
            - Requires matplotlib and networkx for visualization.
"""
from TSPSolver.TSPTour import TSPTour
from TSPData.TSPInstance import TSPInstance
# read max stops from .env if available
try:
    from dotenv import load_dotenv
    from os import getenv
    load_dotenv()
    MAX_STOPS = int(getenv('MAX_STOPS', '20'))
except Exception:
    MAX_STOPS = 20  # Maximum number of stops to visualize clearly


class TSPSolver:
    """Lightweight orchestrator for TSP instances. Actual solver methods
    live in modules under TSPSolver.methods and are selected via the
    registry. This class keeps instance loading and visualization.
    """

    def __init__(self, data: TSPInstance):
        self._data = data
        self._file_name = self._data.file_name

    def Solve(self, **kwargs) -> TSPTour:
        """Dispatch to the requested solver from TSPSolver.methods.registry.

        Expected kwargs:
        - method: name of the solver (registry key); defaults to 'christofides'
        - max_time: optional float seconds (infinity by default)
        """
        # Import registry lazily to avoid circular imports between TSPSolver and Methods
        from TSPSolver.Methods import registry
        method = kwargs.get('method', 'christofides')
        if method not in registry:
            method = 'christofides'
        max_time = float(kwargs.get('max_time', float('inf')))

        SolverClass = registry[method]
        solver_instance = SolverClass(self._data)
        # All solver adapters use solve(max_time)
        return solver_instance.solve(max_time)

    def __del__(self) -> None:
        del self._data

    def Visualisation(self, tour: TSPTour) -> None:
        """ Draw the tour using networkX, if the coordinates are available """
        # Defensive checks: ensure we have a valid tour object
        if tour is None:
            print("No tour provided to Visualisation(). Skipping drawing.")
            return
        if not hasattr(tour, 'sequence') or not hasattr(tour, 'cost'):
            print("Provided object is not a TSPTour (missing attributes). Skipping drawing.")
            return

        if self._data.stops_count > MAX_STOPS:
            print(f"Drawing skipped: too many stops (> {MAX_STOPS}) to visualize clearly.")
            return

        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as e:
            print(
                "matplotlib and networkx are required for the 'Draw' method. Install with "
                f"`pip install matplotlib networkx` or more info see https://matplotlib.org and https://networkx.org. "
                f"Original error: {e}"
            )
            return

        n = self._data.stops_count
        graph = nx.DiGraph()
        pos = {}
        labels = {}
        try:
            for i in range(n):
                graph.add_node(i)
                loc = self._data.coordinates[i]
                px, py = loc.latitude, loc.longitude
                pos[i] = (px, py)
                labels[i] = str(i + 1)
        except Exception as e:
            print(f"Error preparing coordinates for drawing: {e}")
            return
        seq = tour.sequence
        for i in range(n):
            graph.add_edge(seq[i], seq[(i + 1) % n])
        plt.figure(figsize=(8, 8))
        nx.draw(graph, pos, with_labels=True, labels=labels, node_size=300, node_color='lightblue', font_size=10,
                font_weight='bold', arrowsize=15)
        plt.title(f"TSP Tour: {self._file_name.split('\\')[-1]} , cost = {tour.cost:.2f}, method = {tour.method}")
        # Set the window title if possible (matplotlib >= 3.4)
        try:
            fig = plt.gcf()
            if hasattr(fig.canvas.manager, 'set_window_title'):
                fig.canvas.manager.set_window_title(f"TSP Tour: {self._file_name.split('\\')[-1]} , cost = {tour.cost:.2f}, method = {tour.method}")
        except Exception as e:
            print(f"Could not set window title: {e}")
        plt.show()
