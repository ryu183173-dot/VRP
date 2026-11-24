import copy
import random
import time

import numpy as np
from scipy.spatial.distance import euclidean

from configurations import config

(
    battery_capacity,
    vehicle_capacity,
    vehicle_velocity,
    vehicle_energy_decay,
    energy_consumption_per_distance
) = config()


# ==================================================
# Distance calculation
# ==================================================

def compute_distances(clients):

    distances = {}

    for from_client in clients.keys():

        for to_client in clients.keys():

            distances[(from_client, to_client)] = float(
                euclidean(
                    clients[from_client],
                    clients[to_client]
                )
            )

    return distances


def tour_distance(tour, dist_matrix):

    if len(tour) < 2:
        return 0.0

    total_distance = 0.0

    for i in range(1, len(tour)):

        total_distance += dist_matrix[
            (tour[i - 1], tour[i])
        ]

    return float(total_distance)


# ==================================================
# Route decomposition
# ==================================================

def routes_decompose(solution, depot):

    routes = []
    current_route = []

    for node in solution:

        if node == depot:

            if current_route:
                routes.append(current_route)
                current_route = []

        else:
            current_route.append(node)

    if current_route:
        routes.append(current_route)

    return routes


def routes_compose(routes, depot):

    solution = []

    for route in routes:

        if not route:
            continue

        solution.extend(
            [depot] + list(route) + [depot]
        )

    return solution


# ==================================================
# Capacity constraint
# ==================================================

def routes_breaking(
    solution,
    demands,
    max_cap,
    depot
):
    """
    Split routes whenever the vehicle capacity
    would be exceeded.

    The demand dictionary should contain the actual
    demand values of the evaluated scenario.
    """

    if max_cap <= 0:
        raise ValueError("Vehicle capacity must be positive.")

    original_routes = routes_decompose(
        solution,
        depot
    )

    new_routes = []

    for route in original_routes:

        current_route = []
        current_load = 0.0

        for client in route:

            demand = float(demands[client])

            if demand > max_cap + 1e-8:

                raise ValueError(
                    f"Customer {client} demand ({demand}) "
                    f"exceeds vehicle capacity ({max_cap})."
                )

            if (
                current_route
                and current_load + demand > max_cap + 1e-8
            ):

                new_routes.append(current_route)

                current_route = []
                current_load = 0.0

            current_route.append(client)
            current_load += demand

        if current_route:
            new_routes.append(current_route)

    return routes_compose(
        new_routes,
        depot
    )


# ==================================================
# 2-opt
# ==================================================

def two_opt(route, i, j):

    if len(route) < 4:
        return list(route)

    new_route = list(route)

    new_route[i:j + 1] = reversed(
        new_route[i:j + 1]
    )

    return new_route


# ==================================================
# Dynamic travel information
# ==================================================

def get_dynamic_arc_values(
    from_node,
    to_node,
    dist_matrix,
    scenario=None
):

    distance = float(
        dist_matrix[(from_node, to_node)]
    )

    if scenario is None:

        travel_factor = 1.0
        energy_factor = 1.0

    else:

        travel_factor = float(
            scenario['travel_factors'].get(
                (from_node, to_node),
                1.0
            )
        )

        energy_factor = float(
            scenario['energy_factors'].get(
                (from_node, to_node),
                1.0
            )
        )

    travel_time = (
        distance
        / max(float(vehicle_velocity), 1e-8)
        * travel_factor
    )

    energy_used = (
        distance
        * energy_consumption_per_distance
        * energy_factor
    )

    return (
        distance,
        travel_time,
        energy_used
    )


# ==================================================
# Time-window and dynamic route evaluation
# ==================================================

def route_time_eval(
    tour,
    distance_matrix,
    C_TW,
    C_service,
    alpha=0.1,
    beta=0.2,
    scenario=None,
    depot=None,
    route_max_time=None
):
    """
    Evaluate time-window penalties using the actual
    service times and dynamic travel-time factors.

    alpha: waiting-time coefficient
    beta:  lateness penalty coefficient

    Returns a cost value, not a reward.
    """

    if len(tour) < 2:
        return 0.0

    current_time = 0.0
    total_waiting = 0.0
    total_lateness = 0.0

    route_start = 0.0

    for i in range(1, len(tour)):

        from_node = tour[i - 1]
        to_node = tour[i]

        _, travel_time, _ = get_dynamic_arc_values(
            from_node,
            to_node,
            distance_matrix,
            scenario
        )

        current_time += travel_time

        # Depot nodes do not have customer time windows.
        if to_node == depot:
            continue

        if to_node not in C_TW:
            continue

        tw_start = float(
            C_TW[to_node][0]
        )

        tw_end = float(
            C_TW[to_node][1]
        )

        waiting_time = max(
            0.0,
            tw_start - current_time
        )

        total_waiting += waiting_time

        current_time += waiting_time

        lateness = max(
            0.0,
            current_time - tw_end
        )

        total_lateness += lateness

        current_time += float(
            C_service[to_node]
        )

    if route_max_time is not None:

        if current_time - route_start > route_max_time:

            total_lateness += (
                current_time - route_start - route_max_time
            )

    return float(
        alpha * total_waiting
        + beta * total_lateness
    )


# ==================================================
# Dynamic route energy feasibility
# ==================================================

def route_energy_violation(
    tour,
    distance_matrix,
    scenario=None
):
    """
    Each route is assumed to start from a depot
    with a fully charged battery.

    This checks the energy required for the route,
    including the return to the depot.
    """

    total_energy = 0.0

    for i in range(1, len(tour)):

        from_node = tour[i - 1]
        to_node = tour[i]

        _, _, energy_used = get_dynamic_arc_values(
            from_node,
            to_node,
            distance_matrix,
            scenario
        )

        total_energy += energy_used

    return max(
        0.0,
        total_energy - float(battery_capacity)
    )


# ==================================================
# Objective function
# ==================================================

def route_objective(
    solution,
    depot,
    demands,
    max_cap,
    distance_matrix,
    TW_C,
    ServiceT_C,
    alpha=0.1,
    beta=0.2,
    scenario=None,
    route_max_time=None
):
    """
    Common objective for initial and VNS-improved routes.

    Objective:
        total distance
        + time-window penalty
        + energy infeasibility penalty
    """

    # Capacity splitting is performed with actual demand.
    feasible_solution = routes_breaking(
        solution,
        demands,
        max_cap,
        depot
    )

    routes = routes_decompose(
        feasible_solution,
        depot
    )

    total_cost = 0.0

    for route in routes:

        complete_route = (
            [depot] + list(route) + [depot]
        )

        distance_cost = tour_distance(
            complete_route,
            distance_matrix
        )

        time_cost = route_time_eval(
            complete_route,
            distance_matrix,
            TW_C,
            ServiceT_C,
            alpha=alpha,
            beta=beta,
            scenario=scenario,
            depot=depot,
            route_max_time=route_max_time
        )

        energy_violation = route_energy_violation(
            complete_route,
            distance_matrix,
            scenario
        )

        # Strong penalty for a route that exceeds
        # the available battery capacity.
        energy_penalty = (
            1e6 * energy_violation
        )

        total_cost += (
            distance_cost
            + time_cost
            + energy_penalty
        )

    return float(total_cost)


# ==================================================
# Local search
# ==================================================

def local_search(
    route,
    depot,
    demands,
    max_cap,
    distance_matrix,
    TW_C,
    ServiceT_C,
    alpha=0.1,
    beta=0.2,
    scenario=None,
    route_max_time=None
):

    current_solution = routes_breaking(
        route,
        demands,
        max_cap,
        depot
    )

    current_routes = routes_decompose(
        current_solution,
        depot
    )

    improved_routes = []

    for original_route in current_routes:

        best_route = list(original_route)

        if len(best_route) < 3:

            improved_routes.append(best_route)
            continue

        best_solution = routes_compose(
            [best_route],
            depot
        )

        best_cost = route_objective(
            best_solution,
            depot,
            demands,
            max_cap,
            distance_matrix,
            TW_C,
            ServiceT_C,
            alpha,
            beta,
            scenario,
            route_max_time
        )

        improved = True

        while improved:

            improved = False

            for i in range(len(best_route) - 1):

                for j in range(i + 1, len(best_route)):

                    candidate_route = two_opt(
                        best_route,
                        i,
                        j
                    )

                    candidate_solution = routes_compose(
                        [candidate_route],
                        depot
                    )

                    candidate_cost = route_objective(
                        candidate_solution,
                        depot,
                        demands,
                        max_cap,
                        distance_matrix,
                        TW_C,
                        ServiceT_C,
                        alpha,
                        beta,
                        scenario,
                        route_max_time
                    )

                    if candidate_cost + 1e-8 < best_cost:

                        best_route = candidate_route
                        best_cost = candidate_cost
                        improved = True

                        break

                if improved:
                    break

        improved_routes.append(best_route)

    return routes_compose(
        improved_routes,
        depot
    )


# ==================================================
# Shaking
# ==================================================

def shaking(
    route,
    depot,
    demands,
    max_cap,
    distance_matrix,
    TW_C,
    ServiceT_C,
    k,
    operator=two_opt,
    alpha=0.1,
    beta=0.2,
    scenario=None,
    route_max_time=None
):

    routes = routes_decompose(
        routes_breaking(
            route,
            demands,
            max_cap,
            depot
        ),
        depot
    )

    if not routes:
        return [depot, depot]

    for _ in range(max(1, int(k))):

        eligible_routes = [
            index
            for index, item in enumerate(routes)
            if len(item) >= 2
        ]

        if not eligible_routes:
            break

        route_index = random.choice(
            eligible_routes
        )

        selected_route = routes[route_index]

        i, j = sorted(
            random.sample(
                range(len(selected_route)),
                2
            )
        )

        routes[route_index] = operator(
            selected_route,
            i,
            j
        )

    shaken_solution = routes_compose(
        routes,
        depot
    )

    return routes_breaking(
        shaken_solution,
        demands,
        max_cap,
        depot
    )


# ==================================================
# Variable Neighborhood Search
# ==================================================

def vns(
    route,
    depot,
    Demands,
    Max_VC,
    distance_matrix,
    TW_C,
    ServiceT_C,
    k_max=50,
    operator=two_opt,
    scenario=None,
    route_max_time=None,
    alpha=0.1,
    beta=0.2,
    max_runtime=900
):
    """
    VNS with dynamic scenario evaluation.

    The original positional arguments are preserved.
    Additional dynamic arguments are optional.
    """

    start_time = time.time()

    # Normalize and split the initial solution.
    current_solution = routes_breaking(
        route,
        Demands,
        Max_VC,
        depot
    )

    best_solution = copy.deepcopy(
        current_solution
    )

    best_cost = route_objective(
        best_solution,
        depot,
        Demands,
        Max_VC,
        distance_matrix,
        TW_C,
        ServiceT_C,
        alpha,
        beta,
        scenario,
        route_max_time
    )

    k = 1

    exploration_time = 0.0
    exploitation_time = 0.0

    while (
        k <= k_max
        and time.time() - start_time < max_runtime
    ):

        # ------------------------------------------
        # Exploration: shaking
        # ------------------------------------------

        exploration_start = time.time()

        candidate_solution = shaking(
            best_solution,
            depot,
            Demands,
            Max_VC,
            distance_matrix,
            TW_C,
            ServiceT_C,
            k,
            operator=operator,
            alpha=alpha,
            beta=beta,
            scenario=scenario,
            route_max_time=route_max_time
        )

        exploration_time += (
            time.time() - exploration_start
        )

        # ------------------------------------------
        # Exploitation: local search
        # ------------------------------------------

        exploitation_start = time.time()

        candidate_solution = local_search(
            candidate_solution,
            depot,
            Demands,
            Max_VC,
            distance_matrix,
            TW_C,
            ServiceT_C,
            alpha=alpha,
            beta=beta,
            scenario=scenario,
            route_max_time=route_max_time
        )

        exploitation_time += (
            time.time() - exploitation_start
        )

        candidate_solution = routes_breaking(
            candidate_solution,
            Demands,
            Max_VC,
            depot
        )

        candidate_cost = route_objective(
            candidate_solution,
            depot,
            Demands,
            Max_VC,
            distance_matrix,
            TW_C,
            ServiceT_C,
            alpha,
            beta,
            scenario,
            route_max_time
        )

        # ------------------------------------------
        # Acceptance
        # ------------------------------------------

        if candidate_cost + 1e-8 < best_cost:

            best_solution = copy.deepcopy(
                candidate_solution
            )

            best_cost = candidate_cost

            k = 1

        else:

            k += 1

    return (
        best_solution,
        exploration_time,
        exploitation_time
    )