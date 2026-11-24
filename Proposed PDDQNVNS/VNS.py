from scipy.spatial import distance
import numpy as np
import random
import time
import copy


def compute_distances(clients):
    """
    Calculate Euclidean distances between all pairs of nodes.
    """
    distances = {}

    for from_client in clients.keys():
        for to_client in clients.keys():
            distances[(from_client, to_client)] = distance.euclidean(
                clients[from_client],
                clients[to_client]
            )

    return distances


def tour_distance(tour, dist_matrix):
    """
    Calculate the total distance of a route or concatenated routes.
    """
    if len(tour) < 2:
        return 0.0

    total_distance = 0.0

    for i in range(1, len(tour)):
        total_distance += dist_matrix[(tour[i - 1], tour[i])]

    return total_distance


def routes_decompose(solution, depot):
    """
    Split a concatenated solution into individual depot-to-depot routes.

    Example:
        [D, 1, 2, D, D, 3, 4, D]
        ->
        [[D, 1, 2, D], [D, 3, 4, D]]

    Empty routes caused by consecutive depot nodes are ignored.
    """
    routes = []
    current_route = [depot]

    for node in solution[1:]:
        if node == depot:
            if len(current_route) > 1:
                current_route.append(depot)
                routes.append(current_route)

            current_route = [depot]

        else:
            current_route.append(node)

    # Handle a route that does not explicitly end at the depot.
    if len(current_route) > 1:
        current_route.append(depot)
        routes.append(current_route)

    return routes


def routes_combine(routes, depot):
    """
    Combine individual routes into the original flattened representation.

    Consecutive depot nodes separate different routes.
    """
    solution = []

    for route in routes:
        if len(route) <= 2:
            continue

        if solution:
            solution.append(depot)

        solution.extend(route)

    return solution


def two_opt(route, i, j):
    """
    Apply a 2-opt reversal while keeping depot endpoints unchanged.

    i and j are indices in the original route.
    """
    if len(route) <= 3:
        return route[:]

    if i <= 0 or j >= len(route) - 1 or i >= j:
        return route[:]

    new_route = (
        route[:i]
        + route[i:j + 1][::-1]
        + route[j + 1:]
    )

    return new_route


def _two_opt_candidates(route):
    """
    Generate valid 2-opt index pairs.
    """
    candidates = []

    for i in range(1, len(route) - 2):
        for j in range(i + 2, len(route) - 1):
            candidates.append((i, j))

    return candidates


def shaking(tour, k):
    """
    Randomly modify each depot-to-depot route using 2-opt.

    The number of random 2-opt operations is controlled by k.
    Route boundaries are preserved.
    """
    if len(tour) <= 2:
        return tour[:]

    depot = tour[0]
    routes = routes_decompose(tour, depot)

    shaken_routes = []

    for route in routes:
        new_route = route[:]

        for _ in range(k):
            candidates = _two_opt_candidates(new_route)

            if not candidates:
                break

            i, j = random.choice(candidates)
            new_route = two_opt(new_route, i, j)

        shaken_routes.append(new_route)

    return routes_combine(shaken_routes, depot)


def _route_cost(
    tour,
    dist_matrix,
    C_TW=None,
    C_service=None,
    alpha=0.1,
    beta=0.2,
    dist_rate=1,
    time_rate=1
):
    """
    Calculate the combined distance and time-window cost.
    """
    distance_cost = dist_rate * tour_distance(tour, dist_matrix)

    time_cost = 0.0

    if C_TW is not None and C_service is not None:
        time_cost = time_rate * route_time_eval(
            tour,
            dist_matrix,
            C_TW,
            C_service,
            alpha,
            beta
        )

    return distance_cost + time_cost


def local_search(
    tour,
    dist_matrix,
    operator,
    TW=None,
    ST=None,
    alpha=0.1,
    beta=0.2,
    dist_rate=1,
    time_rate=1
):
    """
    Improve each individual route through 2-opt local search.

    The same combined objective used by the VNS acceptance step
    is used when deciding whether to accept a local improvement.
    """
    if len(tour) <= 2:
        return tour[:]

    depot = tour[0]
    routes = routes_decompose(tour, depot)

    improved_routes = []

    for route in routes:
        current_route = route[:]
        improvement_found = True

        while improvement_found:
            improvement_found = False

            candidates = _two_opt_candidates(current_route)

            for i, j in candidates:
                new_route = operator(current_route, i, j)

                current_cost = _route_cost(
                    current_route,
                    dist_matrix,
                    TW,
                    ST,
                    alpha,
                    beta,
                    dist_rate,
                    time_rate
                )

                new_cost = _route_cost(
                    new_route,
                    dist_matrix,
                    TW,
                    ST,
                    alpha,
                    beta,
                    dist_rate,
                    time_rate
                )

                if new_cost < current_cost:
                    current_route = new_route
                    improvement_found = True
                    break

        improved_routes.append(current_route)

    return routes_combine(improved_routes, depot)


def route_time_eval(
    tour,
    distance_matrix,
    C_TW,
    C_service,
    alpha=0.1,
    beta=0.2
):
    """
    Calculate early and late time-window violation penalties.

    Arrival time is calculated from:
        previous departure time + travel time

    If a vehicle arrives before the opening time:
        - Early-arrival penalty is calculated using the paper's formula.
        - The vehicle waits until the opening time before service.

    Service time is added once per customer.
    """
    if len(tour) <= 2:
        return 0.0

    routes = routes_decompose(tour, tour[0])

    total_penalty = 0.0

    for route in routes:
        if len(route) <= 2:
            continue

        current_time = 0.0
        previous_node = route[0]

        for customer in route[1:-1]:

            travel_time = distance_matrix[
                (previous_node, customer)
            ]

            arrival_time = current_time + travel_time

            if customer in C_TW:
                ready_time = C_TW[customer][0]
                due_time = C_TW[customer][1]

                # Early-arrival penalty.
                if arrival_time < ready_time:
                    total_penalty += alpha * (
                        ready_time - arrival_time
                    )

                # Late-arrival penalty.
                elif arrival_time > due_time:
                    total_penalty += beta * (
                        arrival_time - due_time
                    )

                # Wait until the time window opens if arriving early.
                service_start_time = max(
                    arrival_time,
                    ready_time
                )

            else:
                service_start_time = arrival_time

            service_time = C_service.get(customer, 0.0)

            # Service time is added exactly once.
            current_time = service_start_time + service_time

            previous_node = customer

    return total_penalty


def routes_breaking(solution, demands, max_cap, depot):
    """
    Split routes whenever adding the next customer would exceed
    the maximum vehicle capacity.

    The input solution is not modified.
    """
    routes = routes_decompose(solution, depot)

    split_routes = []

    for route in routes:
        current_route = [depot]
        current_demand = 0.0

        for customer in route[1:-1]:

            customer_demand = demands.get(customer, 0.0)

            if (
                current_demand + customer_demand > max_cap
                and len(current_route) > 1
            ):
                current_route.append(depot)
                split_routes.append(current_route)

                current_route = [depot]
                current_demand = 0.0

            current_route.append(customer)
            current_demand += customer_demand

        if len(current_route) > 1:
            current_route.append(depot)
            split_routes.append(current_route)

    return routes_combine(split_routes, depot)


def vns(
    tour,
    depot,
    demands,
    max_cap,
    dist_matrix,
    TW,
    ST,
    alpha=0.1,
    beta=0.2,
    dist_rate=1,
    time_rate=1,
    k_max=50,
    operator=two_opt
):
    """
    Modified Variable Neighborhood Search.

    The stopping condition follows the paper:
        k <= k_max

    Each route is processed independently, while the total
    solution is evaluated using the combined objective.
    """
    k = 1

    total_exploration_time = 0.0
    total_exploitation_time = 0.0

    # Ensure the input is not modified.
    tour = copy.deepcopy(tour)

    # Split routes at capacity violations before optimization.
    tour = routes_breaking(
        tour,
        demands,
        max_cap,
        depot
    )

    while k <= k_max:

        # -------------------------
        # Shaking
        # -------------------------
        time_start = time.time()

        k_tour = shaking(tour, k)

        time_end = time.time()
        total_exploration_time += time_end - time_start

        # -------------------------
        # Local Search
        # -------------------------
        time_start = time.time()

        new_tour = local_search(
            k_tour,
            dist_matrix,
            operator,
            TW,
            ST,
            alpha,
            beta,
            dist_rate,
            time_rate
        )

        time_end = time.time()
        total_exploitation_time += time_end - time_start

        # -------------------------
        # Capacity constraint
        # -------------------------
        new_tour = routes_breaking(
            new_tour,
            demands,
            max_cap,
            depot
        )

        # -------------------------
        # Objective evaluation
        # -------------------------
        cost_new_tour = _route_cost(
            new_tour,
            dist_matrix,
            TW,
            ST,
            alpha,
            beta,
            dist_rate,
            time_rate
        )

        cost_tour = _route_cost(
            tour,
            dist_matrix,
            TW,
            ST,
            alpha,
            beta,
            dist_rate,
            time_rate
        )

        # -------------------------
        # Neighborhood change
        # -------------------------
        if cost_new_tour < cost_tour:
            tour = new_tour
            k = 1

        else:
            k += 1

    return (
        tour,
        total_exploration_time,
        total_exploitation_time
    )
