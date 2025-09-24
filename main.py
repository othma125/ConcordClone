import os
from Data.InputData import input_data  # import the class
from Solver.Tour import tour

if __name__ == "__main__":
    # Use the directory containing this file as project root (was going one level up before)
    project_root = os.path.dirname(os.path.abspath(__file__))
    tsp_dir = os.path.join(project_root, "ALL_tsp")

    # file_name = "burma14.tsp"  # change to desired file
    file_name = "bier127.tsp"
    selected_file = os.path.join(tsp_dir, file_name)
    if not os.path.isfile(selected_file):
        raise FileNotFoundError(f"TSP file not found: {selected_file}")

    data = input_data(selected_file)
    print(data)
    # print(data.get_cost(0, 1))
    # print(data.get_cost(1, 2))
    route = tour(data)
    print(route)
