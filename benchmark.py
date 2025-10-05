from TSPData.TSPInstance import TSPInstance
from TSPSolver.TSPSolver import TSPSolver
from TSPSolver.TSPTour import TSPTour
import pandas as pd

df = pd.read_csv("DefaultInstances//TSPLIB//best_known_values.csv")


def calculate_gap(file_name: str, route: TSPTour, df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the gap between the best known solution and the current solution"""
    
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
              'method_value': float(route.cost),
              'method_time(s)': round(route.reach_time, 2),
              'method_name': route.solution_methode,
              'stops_count': len(route.sequence),
              'gap': f"{gap:.2%}"}
    return result


if __name__ == "__main__":
    max_dimension = 14
    # loop over all .tsp files in the TSPLIB folder and store it in a list
    from pathlib import Path
    # benchmark.py sits at the repository root, so use its parent as repo_root
    repo_root = Path(__file__).resolve().parent
    tsp_dir = repo_root / "DefaultInstances" / "TSPLIB"
    # glob('*.tsp') already restricts to .tsp files; use the Path objects
    # and pass a string path to TSPInstance to avoid Path-related issues.
    files = {file: TSPInstance(str(file)).stops_count for file in tsp_dir.glob("*.tsp")}
    files = {file: dim for file, dim in files.items() if dim <= max_dimension}
    # sort the files by the number of stops and produce (file, dim) tuples
    files_list = sorted(files.items(), key=lambda kv: kv[1])
    from TSPSolver.TSPSolver import methodes
    from pprint import pprint
    results = []
    for file, dim in files_list:
        # print(f"Solving {file.name} with {dim} stops")
        solver = TSPSolver(str(file))
        for method in methodes.keys():
            features = {
                'method': method,
                'max_time': 10}  # seconds
            route = solver.Solve(**features)
            result = calculate_gap(file.name, route, df)
            results.append(result)
            print("Result: ")
            pprint(result)
            print()
    columns = 'file_name,stops_count,best_known_value,best_known_time(s),method_name,method_value,method_time(s),gap'.split(',')
    output_file = repo_root / f"benchmark_results_up_to_{max_dimension}_stops.csv"
    # Build DataFrame from results. If empty, create an empty DataFrame with
    # the desired columns so the CSV still contains headers.
    df_out = pd.DataFrame(results)
    if df_out.empty:
        df_out = pd.DataFrame(columns=columns)
    else:
        # Reorder/add missing columns to match our desired schema
        df_out = df_out.reindex(columns=columns)

    df_out.to_csv(output_file, index=False)
    print(f"Wrote benchmark results to: {output_file} ({len(df_out)} rows)")
