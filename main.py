from TSPSolver.TSPSolver import TSPSolver
from TSPSolver.TSPTour import TSPTour


if __name__ == "__main__":
    # file_name = "bier127.tsp"
    file_name = "burma14.tsp"

    solver = TSPSolver(file_name)
    features = {
        # 'method' : "christofides"
        # 'method' : "nearest_neighbor"
        # 'method': "chained_LK"
        # 'method' : "pyvrp_hgs"
        'method' : "concord_wrapper"
        , 'max_time': 10}  # seconds
    route = solver.Solve(**features)
    print(route)
    solver.Visualisation(route)
