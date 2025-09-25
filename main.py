from Solver.Solver import Solver

if __name__ == "__main__":
    import sys
    print(sys.version)
    file_name = "bier127.tsp"
    # file_name = "burma14.tsp"

    solver = Solver(file_name)
    features = {
                'method' : "chained_LK"
                # 'method' : "nearest_neighbor"
                , 'max_time' : 30} # seconds
    route = solver.Solve(**features)  
    print(route)
