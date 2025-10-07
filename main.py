from TSPSolver.TSPSolver import TSPSolver
from TSPData.TSPInstance import TSPInstance

if __name__ == "__main__":
    ## 1) Example: read from TSPLIB file (uncomment to test with a local file)
    # file_name = "bier127.tsp"
    file_name = "burma14.tsp"
    from pyparsing import Path

    repo_root = Path(__file__).resolve().parent
    selected_file = repo_root / "DefaultInstances" / "TSPLIB" / file_name
    if not selected_file.is_file():
        raise FileNotFoundError(f"TSP file not found: {selected_file}")
    data = TSPInstance(selected_file)

    ## 2) Example: construct from coordinates (small square)
    # coords = [(0.10, 0.0), (1.50, 0.0), (1.30, 1.0), (0.0, 1.0)]
    # data = TSPInstance(coordinates=coords)

    ## 3) Example: construct from explicit distance matrix
    # import numpy as np
    # m = np.array([[0,1,2],[1,0,1],[2,1,0]], dtype=float)
    # data = TSPInstance(matrix=m)

    solver = TSPSolver(data)
    features = {
        # 'method': "christofides"
        # 'method' : "nearest_neighbor"
        # 'method': "chained_LK"
        # 'method' : "pyvrp_hgs"
        # 'method' : "concord_wrapper"
        'method': "simulated_annealing"
        , 'max_time': 4}  # seconds
    route = solver.Solve(**features)
    print(route)
    solver.Visualisation(route)
