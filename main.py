from Solver.Solver import Solver

if __name__ == "__main__":
    # file_name = "bier127.tsp"
    file_name = "burma14.tsp"

    solver = Solver(file_name)
    features = {
                # 'method' : "nearest_neighbor"
                # 'method' : "christofides"
                'method' : "chained_LK"
                # 'method' : "concord_wrapper"
                , 'max_time' : 10} # seconds
    route = solver.Solve(**features)  
    print(route)
