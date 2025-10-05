from TSPSolver.Methods.TSPSolver import TSPSolver
from TSPData.TSPInstance import TSPInstance
from pathlib import Path


if __name__ == "__main__":
    # file_name = "bier127.tsp"
    file_name = "burma14.tsp"
    repo_root = Path(__file__).resolve().parent
    selected_file = repo_root / "DefaultInstances" / "TSPLIB" / file_name
    if not selected_file.is_file():
        raise FileNotFoundError(f"TSP file not found: {selected_file}")
    # Construct solver with file name (TSPSolver will parse the instance)
    data = TSPInstance(selected_file)
    solver = TSPSolver(data)
    features = {
        'method' : "christofides"
        # 'method' : "nearest_neighbor"
        # 'method': "chained_LK"
        # 'method' : "pyvrp_hgs"
        # 'method' : "concord_wrapper"
        , 'max_time': 10}  # seconds
    route = solver.Solve(**features)
    print(route)
    solver.Visualisation(route)
