from TSPSolver.TSPSolver import TSPSolver
from TSPSolver.TSPTour import tour
from pandas import DataFrame


def calculate_gap(file_name: str, route: tour) -> DataFrame:
    import pandas as pd
    df = pd.read_csv("TSPLIB//best_known_values.csv")
    # Calculate the gap between the best known solution and the current solution
    instance_name = file_name.split('.')[0]
    best_known_row = df[df['Instance'] == instance_name]
    if best_known_row.empty:
        raise ValueError(f"No best known solution found for instance: {instance_name}")
    best_known_value = best_known_row['BestKnownValue'].values[0]
    best_known_time = best_known_row['RunningTime'].values[0]
    gap = (float(route.cost) - best_known_value) / best_known_value if best_known_value != 0 else 0
    # Create a dictionary as result
    result = {'file_name': file_name,
              'best_known_value': float(best_known_value),
              'best_known_time(s)': best_known_time,
              'current_value': float(route.cost),
              'current_time(s)': route.reach_time,
              'gap': f"{gap:.2%}"}
    return result


if __name__ == "__main__":
    file_name = "bier127.tsp"
    # file_name = "burma14.tsp"

    solver = TSPSolver(file_name)
    features = {
        'method' : "christofides"
        # 'method' : "nearest_neighbor"
        # 'method': "chained_LK"
        # 'method' : "concord_wrapper"
        , 'max_time': 10}  # seconds
    route = solver.Solve(**features)
    print(route)
    from pprint import pprint
    pprint(calculate_gap(file_name, route))
