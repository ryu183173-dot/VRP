from Reading_dataset import *
from model_functions import *
from scipy.spatial.distance import euclidean
from Algorithm import RL_Agent
from VNS import *
from tqdm import tqdm

import os
import ast
import time
import numpy as np
import pandas as pd


# 1回の実行で新たに処理するインスタンス数
MAX_INSTANCES_PER_RUN = 1


if __name__ == "__main__":

    dataset = "Vidal"
    csv_path = "./PAVNS_Vidal.csv"

    Total_Solution = {}
    Total_Distances = {}
    RunTime = {}

    # --------------------------------------------------
    # 1. Load previously completed results
    # --------------------------------------------------

    if os.path.exists(csv_path):

        results_df = pd.read_csv(csv_path)

        completed_instances = set(
            results_df["DS"].dropna().astype(str)
        )

        results = results_df.to_dict(orient="records")

        for _, row in results_df.iterrows():

            ds = str(row["DS"])

            Total_Solution[ds] = ast.literal_eval(
                str(row["Final_Solution"])
            )

            Total_Distances[ds] = {
                "Initial Solution": float(row["Initial_Dist"]),
                "Sum": float(row["Final_Dist"]),
                "Cust_Num": int(row["Cust_Num"]),
                "Depot_Num": int(row["Depot_Num"])
            }

            RunTime[ds] = float(row["Time"])

        print("保存済みインスタンス数:", len(completed_instances))
        print("保存済みインスタンス:", sorted(completed_instances))

    else:

        completed_instances = set()
        results = []

        print("保存済みCSVがないため、最初から実行します。")

    # --------------------------------------------------
    # 2. Instance settings
    # --------------------------------------------------

    processed_this_run = 0

    file_id = list(range(11, 25))
    dataset_subs = ["a", "b"]

    for dataset_sub in dataset_subs:

        for dataset_num in tqdm(
            file_id,
            desc=f"Instances ({dataset_sub}): "
        ):

            if processed_this_run >= MAX_INSTANCES_PER_RUN:
                break

            ds_title = f"pr{dataset_num}{dataset_sub}"

            print(
                f"\n============ {ds_title} ============\n"
            )

            # Skip completed instances
            if ds_title in completed_instances:

                print(
                    f"{ds_title} は保存済みのためスキップします。"
                )

                continue

            data_path = (
                "./Dataset/Public/vidal-al-2013-mdvrptw/"
                f"pr{dataset_num}{dataset_sub}.txt"
            )

            # --------------------------------------------------
            # 3. Dataset loading
            # --------------------------------------------------

            data_df, depot_df, Vehicle_info, data_conf = (
                reading_vidal_ds(data_path)
            )

            depot_num = len(depot_df)

            # --------------------------------------------------
            # 4. Data normalization
            # --------------------------------------------------

            data_df["XCOORD."] = (
                data_df["XCOORD."] + 100
            ) / 200

            data_df["YCOORD."] = (
                data_df["YCOORD."] + 100
            ) / 200

            data_df["DEMAND"] = data_df["DEMAND"] / 50

            data_df["SERVICE_TIME"] = (
                data_df["SERVICE_TIME"] / 50
            )

            data_df["READY_TIME"] = (
                data_df["READY_TIME"] / 1000
            )

            data_df["DUE_DATE"] = (
                data_df["DUE_DATE"] / 1000
            )

            depot_df["XCOORD."] = (
                depot_df["XCOORD."] + 100
            ) / 200

            depot_df["YCOORD."] = (
                depot_df["YCOORD."] + 100
            ) / 200

            depot_df["READY_TIME"] = (
                depot_df["READY_TIME"] / 1000
            )

            depot_df["DUE_DATE"] = (
                depot_df["DUE_DATE"] / 1000
            )

            # --------------------------------------------------
            # 5. Construct problem data
            # --------------------------------------------------

            coordinates_customers = {}
            time_windows_customers = {}
            demands_customers = {}
            service_times_customers = {}

            coordinates_depots = {}
            time_windows_depots = {}

            total_coords = {}

            index = 0

            # Depots
            for item in range(depot_num):

                coordinates_depots[index] = [
                    depot_df["XCOORD."].iloc[item],
                    depot_df["YCOORD."].iloc[item]
                ]

                time_windows_depots[index] = [
                    depot_df["READY_TIME"].iloc[item],
                    depot_df["DUE_DATE"].iloc[item]
                ]

                total_coords[index] = (
                    coordinates_depots[index]
                )

                index += 1

            # Customers
            for item in data_df.index:

                coordinates_customers[index] = [
                    data_df.loc[item, "XCOORD."],
                    data_df.loc[item, "YCOORD."]
                ]

                time_windows_customers[index] = [
                    data_df.loc[item, "READY_TIME"],
                    data_df.loc[item, "DUE_DATE"]
                ]

                demands_customers[index] = (
                    data_df.loc[item, "DEMAND"]
                )

                service_times_customers[index] = (
                    data_df.loc[item, "SERVICE_TIME"]
                )

                total_coords[index] = (
                    coordinates_customers[index]
                )

                index += 1

            # --------------------------------------------------
            # 6. DRL settings
            # --------------------------------------------------

            C = coordinates_customers
            D = coordinates_depots

            TW_C = time_windows_customers
            Demands = demands_customers
            ServiceT_C = service_times_customers

            Max_VC = 1

            maxTime = Vehicle_info["max_T"][0] / 10

            Max_VRT = maxTime

            n_episodes = 200
            n_vehicles = depot_num

            n_timesteps = (
                max(data_df["DUE_DATE"]) + 100
            )

            alpha = 0.1
            beta = 0.2

            # --------------------------------------------------
            # 7. Distance matrix and depot clusters
            # --------------------------------------------------

            start_time = time.time()

            distance_matrix = compute_distances(
                total_coords
            )

            clusters = {}

            for depot in D:
                clusters[depot] = []

            # Assign each customer to its nearest depot
            for customer in C.keys():

                nearest_depot = min(
                    D.keys(),
                    key=lambda depot: distance_matrix[
                        (customer, depot)
                    ]
                )

                clusters[nearest_depot].append(customer)

            print("Clusters:", clusters)

            # --------------------------------------------------
            # 8. DDQN
            # --------------------------------------------------

            agent, routes = RL_Agent(
                C,
                D,
                TW_C,
                Demands,
                Max_VC,
                clusters,
                ServiceT_C,
                n_vehicles,
                n_episodes,
                n_timesteps,
                Q_update_frequency=2,
                Target_update_freq=4,
                route_max_time=Max_VRT,
                alpha=alpha,
                beta=beta
            )

            # Retrieve the scenario used in the final episode
            if not hasattr(agent, "last_scenario"):

                raise AttributeError(
                    "RL_Agent does not contain last_scenario. "
                    "Check Algorithm.py."
                )

            scenario = agent.last_scenario

            if scenario is None:

                raise ValueError(
                    "agent.last_scenario is None. "
                    "The final episode scenario was not saved."
                )

            dynamic_demands = scenario["actual_demands"]
            dynamic_service = scenario["actual_service"]

            # --------------------------------------------------
            # 9. Construct initial solution
            #    Keep vehicle routes separate
            # --------------------------------------------------

            initial_solution = {}

            for depot in routes.keys():

                initial_solution[depot] = []

                for vehicle_routes in routes[depot].values():

                    for customer_route in vehicle_routes:

                        if len(customer_route) == 0:
                            continue

                        tour = (
                            [depot]
                            + list(customer_route)
                            + [depot]
                        )

                        initial_solution[depot].append(tour)

            print("\nInitial Solution:")
            print(initial_solution)

            # --------------------------------------------------
            # 10. VNS
            #     Apply VNS to each vehicle route independently
            # --------------------------------------------------

            final_solution = {}

            initial_distance = 0.0
            final_distance = 0.0

            for depot in initial_solution.keys():

                final_solution[depot] = []

                for route in initial_solution[depot]:

                    if len(route) <= 2:
                        continue

                    init_dist = tour_distance(
                        route,
                        distance_matrix
                    )

                    initial_distance += init_dist

                    best_tour, exploration_time, exploitation_time = (
                        vns(
                            route,
                            depot,
                            dynamic_demands,
                            Max_VC,
                            distance_matrix,
                            TW_C,
                            dynamic_service,
                            k_max=50,
                            operator=two_opt,
                            scenario=scenario,
                            route_max_time=Max_VRT,
                            alpha=alpha,
                            beta=beta
                        )
                    )

                    fin_dist = tour_distance(
                        best_tour,
                        distance_matrix
                    )

                    final_distance += fin_dist

                    final_solution[depot].append(
                        best_tour
                    )

            # --------------------------------------------------
            # 11. Results
            # --------------------------------------------------

            print("\nInitial Solution:")
            print(initial_solution)

            print("Initial Distance:", initial_distance)

            print("\nFinal Solution:")
            print(final_solution)

            print("Final Distance:", final_distance)

            Total_Solution[ds_title] = final_solution

            Total_Distances[ds_title] = {
                "Initial Solution": initial_distance,
                "Sum": final_distance,
                "Cust_Num": len(data_df),
                "Depot_Num": len(depot_df)
            }

            finish_time = time.time()

            RunTime[ds_title] = (
                finish_time - start_time
            )

            results.append({
                "DS": ds_title,
                "Initial_Solution": initial_solution,
                "Final_Solution": final_solution,
                "Initial_Dist": initial_distance,
                "Final_Dist": final_distance,
                "Time": RunTime[ds_title],
                "Cust_Num": len(data_df),
                "Depot_Num": len(depot_df)
            })

            print(
                f"\n================ Evaluation For Instance "
                f"{ds_title} Finished ======================\n"
            )

            # Save after each completed instance
            results_df = pd.DataFrame(results)

            results_df.to_csv(
                csv_path,
                index=False
            )

            completed_instances.add(ds_title)

            processed_this_run += 1

            print(
                "今回の実行で完了した数: "
                f"{processed_this_run}/{MAX_INSTANCES_PER_RUN}"
            )

        if processed_this_run >= MAX_INSTANCES_PER_RUN:
            break

    # --------------------------------------------------
    # 12. Final log
    # --------------------------------------------------

    with open("Vidal_PAVNS_Log.txt", "w") as file:

        distances_sum = []
        runtime_sum = []

        for key in Total_Solution.keys():

            print(
                key,
                "Solution:",
                Total_Solution[key]
            )

            print(
                key,
                "Distance:",
                Total_Distances[key]["Sum"]
            )

            distances_sum.append(
                Total_Distances[key]["Sum"]
            )

            print(
                "Run Time:",
                RunTime[key]
            )

            runtime_sum.append(
                RunTime[key]
            )

            print("==========================\n")

            file.write(
                f"{key} >> Solution: {Total_Solution[key]}\n"
            )

            file.write(
                f"{key} >> Distance: "
                f"{Total_Distances[key]['Sum']}\n"
            )

            file.write(
                f"{key} >> Run Time: {RunTime[key]}\n"
            )

            file.write(
                "==========================\n"
            )

        mean_distance = (
            np.mean(distances_sum)
            if distances_sum else 0
        )

        mean_runtime = (
            np.mean(runtime_sum)
            if runtime_sum else 0
        )

        print(
            "Mean Total Sum Distances on the instances:",
            mean_distance
        )

        print(
            "Mean Total Run Time on the instances (s):",
            mean_runtime
        )

        file.write(
            "Mean Total Sum Distances on the instances: "
            f"{mean_distance}\n"
        )

        file.write(
            "Mean Total Run Time on the instances (s): "
            f"{mean_runtime}\n"
        )