from PDQN_Network import *
from Reading_dataset import *
from scipy.spatial.distance import euclidean

import numpy as np
import copy
from tqdm import tqdm

from configurations import config

(
    battery_capacity,
    vehicle_capacity,
    vehicle_velocity,
    vehicle_energy_decay,
    energy_consumption_per_distance
) = config()


# ==================================================
# Energy consumption
# ==================================================

def energy_consumption(dist, current_energy, factor=1.0):
    energy_used = (
        float(dist)
        * float(energy_consumption_per_distance)
        * float(factor)
    )

    return max(0.0, float(current_energy) - energy_used)


# ==================================================
# Position helper
# ==================================================

def get_current_position(env, depot, D):

    current_id = env['current_state_id']

    if current_id in D:
        return tuple(D[current_id])

    return tuple(env['client_coords'][current_id])


# ==================================================
# Candidate-specific 9-dimensional state
# ==================================================

def build_candidate_states(env, depot, D, candidate_ids):

    current_position = get_current_position(
        env,
        depot,
        D
    )

    candidate_states = []

    for client in candidate_ids:

        target_position = env['client_coords'][client]

        tw_start, tw_end = env['client_TW'][client]

        state = np.array(
            [
                current_position[0],
                current_position[1],
                target_position[0],
                target_position[1],
                env['energy'],
                env['Q'],
                env['total_time'],
                tw_start,
                tw_end
            ],
            dtype=np.float32
        )

        candidate_states.append(state)

    if not candidate_states:
        return np.empty((0, 9), dtype=np.float32)

    return np.asarray(
        candidate_states,
        dtype=np.float32
    )


# ==================================================
# Dynamic scenario generation
# ==================================================

def generate_dynamic_scenario(
    C,
    D,
    Demands,
    T_Service,
    scenario_seed,
    max_capacity
):
    """
    Generate one reproducible dynamic scenario.

    Experimental assumptions:
        Demand factor:       U(0.8, 1.2)
        Service-time factor: U(0.8, 1.2)
        Travel-time factor:  U(0.9, 1.3)
        Energy factor:       U(0.9, 1.2)

    Actual demand and service time are stored in the
    environment, but are not used for candidate filtering.
    """

    rng = np.random.default_rng(scenario_seed)

    actual_demands = {}
    actual_service = {}

    for client in C.keys():

        base_demand = float(Demands[client])
        base_service = float(T_Service[client])

        actual_demands[client] = (
            base_demand * rng.uniform(0.8, 1.2)
        )

        actual_service[client] = (
            base_service * rng.uniform(0.8, 1.2)
        )

    # A customer whose actual demand exceeds the full
    # vehicle capacity cannot be served by this vehicle.
    infeasible_clients = [
        client
        for client, demand in actual_demands.items()
        if demand > float(max_capacity) + 1e-8
    ]

    if infeasible_clients:
        raise ValueError(
            "Dynamic scenario contains customers whose "
            "actual demand exceeds the vehicle capacity. "
            f"Customers: {infeasible_clients}"
        )

    all_nodes = list(D.keys()) + list(C.keys())

    travel_factors = {}
    energy_factors = {}

    for from_node in all_nodes:

        for to_node in all_nodes:

            if from_node == to_node:
                continue

            travel_factors[(from_node, to_node)] = (
                float(rng.uniform(0.9, 1.3))
            )

            energy_factors[(from_node, to_node)] = (
                float(rng.uniform(0.9, 1.2))
            )

    return {
        'seed': int(scenario_seed),
        'actual_demands': actual_demands,
        'actual_service': actual_service,
        'travel_factors': travel_factors,
        'energy_factors': energy_factors
    }


# ==================================================
# Travel information
# ==================================================

def get_travel_information(
    from_id,
    to_id,
    from_position,
    to_position,
    scenario
):

    dist = float(
        euclidean(
            from_position,
            to_position
        )
    )

    travel_factor = scenario['travel_factors'].get(
        (from_id, to_id),
        1.0
    )

    energy_factor = scenario['energy_factors'].get(
        (from_id, to_id),
        1.0
    )

    travel_time = (
        dist / max(float(vehicle_velocity), 1e-8)
    )

    travel_time *= float(travel_factor)

    return (
        dist,
        travel_time,
        float(energy_factor)
    )


# ==================================================
# Return-to-depot operation
# ==================================================

def schedule_return_to_depot(
    env,
    depot,
    D,
    scenario
):
    """
    Start a physical return trip.

    Capacity and battery are restored only after
    the vehicle reaches its depot.
    """

    if env['returning_to_depot']:
        return

    current_position = get_current_position(
        env,
        depot,
        D
    )

    depot_position = tuple(D[depot])

    current_id = env['current_state_id']

    dist, travel_time, energy_factor = (
        get_travel_information(
            current_id,
            depot,
            current_position,
            depot_position,
            scenario
        )
    )

    env['total_distance'] += dist

    env['energy'] = energy_consumption(
        dist,
        env['energy'],
        energy_factor
    )

    env['total_time'] += travel_time
    env['current_route_time'] += travel_time

    if env['current_route']:

        env['total_routes'].append(
            list(env['current_route'])
        )

        env['current_route'] = []

    env['returning_to_depot'] = True
    env['idle'] = True


# ==================================================
# Candidate feasibility
# ==================================================

def get_feasible_candidates(
    env,
    depot,
    D,
    candidate_ids,
    Demands,
    T_Service,
    route_max_time
):
    """
    Candidate filtering uses known/base information.

    Actual demand and actual service time are NOT
    exposed to this filtering operation.
    """

    current_position = get_current_position(
        env,
        depot,
        D
    )

    current_id = env['current_state_id']

    depot_position = tuple(D[depot])

    feasible_candidates = []

    for client in candidate_ids:

        target_position = tuple(
            env['client_coords'][client]
        )

        outbound_dist = float(
            euclidean(
                current_position,
                target_position
            )
        )

        return_dist = float(
            euclidean(
                target_position,
                depot_position
            )
        )

        # Conservative energy feasibility.
        conservative_energy = (
            outbound_dist + return_dist
        ) * energy_consumption_per_distance * 1.2

        if env['energy'] + 1e-8 < conservative_energy:
            continue

        # Conservative travel-time feasibility.
        conservative_travel_time = (
            (outbound_dist + return_dist)
            / max(float(vehicle_velocity), 1e-8)
            * 1.3
        )

        # Base service time is used instead of actual service time.
        conservative_service_time = (
            float(T_Service[client]) * 1.2
        )

        # Estimate waiting time using the known time window.
        estimated_arrival = (
            env['total_time']
            + outbound_dist
            / max(float(vehicle_velocity), 1e-8)
            * 1.3
        )

        tw_start = float(
            env['client_TW'][client][0]
        )

        estimated_waiting = max(
            0.0,
            tw_start - estimated_arrival
        )

        estimated_route_time = (
            env['current_route_time']
            + conservative_travel_time
            + conservative_service_time
            + estimated_waiting
        )

        if estimated_route_time > route_max_time:
            continue

        # Prevent selecting a customer whose base demand
        # already exceeds the vehicle's full capacity.
        if float(Demands[client]) > float(env['capacity']) + 1e-8:
            continue

        feasible_candidates.append(client)

    return feasible_candidates


# ==================================================
# Main RL algorithm
# ==================================================

def RL_Agent(
    C,
    D,
    TW_C,
    Demands,
    Q,
    Clusters,
    T_Service,
    n_vehicles,
    n_episodes,
    n_time_steps,
    Q_update_frequency=10,
    Target_update_freq=100,
    route_max_time=500,
    alpha=0.1,
    beta=0.2,
    scenario_seed=42
):

    # Keep the existing 9-dimensional state.
    state_dim = 9

    action_dim = len(C) + len(D)

    # --------------------------------------------------
    # Cluster information
    # --------------------------------------------------

    cluster_TW = {}
    cluster_coords = {}

    for depot in D.keys():

        cluster_TW[depot] = {}
        cluster_coords[depot] = {}

        for client in Clusters[depot]:

            cluster_TW[depot][client] = tuple(
                TW_C[client]
            )

            cluster_coords[depot][client] = tuple(
                C[client]
            )

    # --------------------------------------------------
    # DDQN Agent
    # --------------------------------------------------

    Agents = DoubleDQNAgent(
        gamma=0.99,
        epsilon=1.0,
        batch_size=64,
        n_actions=action_dim,
        eps_end=0.01,
        input_dims=state_dim,
        lr=0.003
    )

    Agents.Q_eval.exploration_proba = 0.5

    # --------------------------------------------------
    # Vehicle assignment
    # --------------------------------------------------

    if n_vehicles < len(D):
        raise ValueError(
            "n_vehicles must be at least the number of depots."
        )

    dv_assignment = {
        depot: []
        for depot in D.keys()
    }

    depot_ids = list(D.keys())

    for vehicle_index in range(n_vehicles):

        depot_index = vehicle_index % len(depot_ids)

        depot = depot_ids[depot_index]

        dv_assignment[depot].append(vehicle_index)

    print("Vehicle assignment:", dv_assignment)

    # --------------------------------------------------
    # Episode training
    # --------------------------------------------------

    total_routes = {}

    Agents.last_scenario = None
    Agents.last_episode_complete = False

    for episode in tqdm(
        range(n_episodes),
        desc="DDQN Dynamic Episodes"
    ):

        # A reproducible scenario for each episode.
        episode_scenario = generate_dynamic_scenario(
            C,
            D,
            Demands,
            T_Service,
            scenario_seed + episode,
            Q
        )

        envs = {}

        # --------------------------------------------------
        # Environment initialization
        # --------------------------------------------------

        for depot in D.keys():

            envs[depot] = {}

            for vehicle_id in dv_assignment[depot]:

                envs[depot][vehicle_id] = {

                    'done': False,

                    'total_routes': [],
                    'current_route': [],

                    'total_time': 0.0,
                    'total_distance': 0.0,
                    'current_route_time': 0.0,

                    'Q': float(Q),
                    'capacity': float(Q),

                    'idle': False,
                    'returning_to_depot': False,

                    # Actual values remain in the environment.
                    'client_demands': copy.deepcopy(
                        episode_scenario['actual_demands']
                    ),

                    'client_TW': copy.deepcopy(
                        cluster_TW[depot]
                    ),

                    'client_coords': copy.deepcopy(
                        cluster_coords[depot]
                    ),

                    'client_service': copy.deepcopy(
                        episode_scenario['actual_service']
                    ),

                    'current_state_id': depot,

                    'current_state': np.zeros(
                        state_dim,
                        dtype=np.float32
                    ),

                    'next_state': np.zeros(
                        state_dim,
                        dtype=np.float32
                    ),

                    'unvisited_clients': list(
                        Clusters[depot]
                    ),

                    'energy': float(battery_capacity)
                }

        global_time_step = 0.0
        update_steps = 0

        all_done = False

        # --------------------------------------------------
        # Main simulation loop
        # --------------------------------------------------

        while (
            not all_done
            and global_time_step <= n_time_steps
        ):

            for depot in D.keys():

                for vehicle_id in dv_assignment[depot]:

                    env = envs[depot][vehicle_id]

                    if env['done']:
                        continue

                    # Vehicle is travelling or serving.
                    if global_time_step + 1e-8 < env['total_time']:
                        continue

                    # --------------------------------------------------
                    # Complete return trip
                    # --------------------------------------------------

                    if env['returning_to_depot']:

                        env['current_state_id'] = depot

                        env['energy'] = float(
                            battery_capacity
                        )

                        env['Q'] = float(
                            env['capacity']
                        )

                        env['current_route_time'] = 0.0

                        env['returning_to_depot'] = False
                        env['idle'] = False

                    # --------------------------------------------------
                    # Completion check
                    # --------------------------------------------------

                    if not env['unvisited_clients']:

                        if env['current_route']:

                            env['total_routes'].append(
                                list(env['current_route'])
                            )

                            env['current_route'] = []

                        env['done'] = True

                        continue

                    # --------------------------------------------------
                    # Candidate list from SAP
                    # --------------------------------------------------

                    candidate_ids = list(
                        env['unvisited_clients']
                    )

                    # --------------------------------------------------
                    # Return decision
                    # --------------------------------------------------

                    current_position = get_current_position(
                        env,
                        depot,
                        D
                    )

                    current_id = env['current_state_id']

                    depot_position = tuple(D[depot])

                    return_dist = float(
                        euclidean(
                            current_position,
                            depot_position
                        )
                    )

                    return_energy_need = (
                        return_dist
                        * energy_consumption_per_distance
                        * 1.2
                    )

                    base_demands_remaining = [
                        float(Demands[client])
                        for client in candidate_ids
                    ]

                    min_base_demand = (
                        min(base_demands_remaining)
                        if base_demands_remaining
                        else 0.0
                    )

                    should_return = (

                        env['Q'] + 1e-8 < min_base_demand

                        or env['current_route_time']
                        >= route_max_time

                        or env['energy'] + 1e-8
                        < return_energy_need
                    )

                    if should_return:

                        schedule_return_to_depot(
                            env,
                            depot,
                            D,
                            episode_scenario
                        )

                        continue

                    # --------------------------------------------------
                    # Feasible candidate filtering
                    # --------------------------------------------------

                    feasible_candidates = get_feasible_candidates(
                        env,
                        depot,
                        D,
                        candidate_ids,
                        Demands,
                        T_Service,
                        route_max_time
                    )

                    if not feasible_candidates:

                        # If no route has been started and no customer
                        # can be served from the depot, the instance
                        # cannot be completed under the current limits.
                        if (
                            not env['current_route']
                            and current_id == depot
                        ):
                            raise RuntimeError(
                                "No feasible customer can be reached "
                                f"from depot {depot}. "
                                "Check route_max_time, battery capacity, "
                                "vehicle capacity and data scaling."
                            )

                        schedule_return_to_depot(
                            env,
                            depot,
                            D,
                            episode_scenario
                        )

                        continue

                    # --------------------------------------------------
                    # Candidate states
                    # --------------------------------------------------

                    candidate_states = build_candidate_states(
                        env,
                        depot,
                        D,
                        feasible_candidates
                    )

                    selected_index = Agents.choose_action(
                        candidate_states
                    )

                    selected_index = int(selected_index)

                    if (
                        selected_index < 0
                        or selected_index >= len(feasible_candidates)
                    ):
                        raise ValueError(
                            "DDQN returned an invalid candidate index."
                        )

                    next_client = feasible_candidates[
                        selected_index
                    ]

                    current_state = candidate_states[
                        selected_index
                    ].copy()

                    # --------------------------------------------------
                    # Travel to customer
                    # --------------------------------------------------

                    next_position = tuple(
                        env['client_coords'][next_client]
                    )

                    dist, travel_time, energy_factor = (
                        get_travel_information(
                            current_id,
                            next_client,
                            current_position,
                            next_position,
                            episode_scenario
                        )
                    )

                    departure_time = float(
                        env['total_time']
                    )

                    arrival_time = (
                        departure_time + travel_time
                    )

                    # --------------------------------------------------
                    # Energy consumption
                    # --------------------------------------------------

                    env['energy'] = energy_consumption(
                        dist,
                        env['energy'],
                        energy_factor
                    )

                    env['total_distance'] += dist

                    env['current_state_id'] = next_client

                    # --------------------------------------------------
                    # Reveal actual demand at arrival
                    # --------------------------------------------------

                    actual_demand = float(
                        env['client_demands'][next_client]
                    )

                    # If actual demand exceeds remaining capacity,
                    # do not serve it. Return and try again after reload.
                    if actual_demand > env['Q'] + 1e-8:

                        env['total_time'] = arrival_time

                        env['current_route_time'] += travel_time

                        schedule_return_to_depot(
                            env,
                            depot,
                            D,
                            episode_scenario
                        )

                        continue

                    # --------------------------------------------------
                    # Time-window evaluation
                    # --------------------------------------------------

                    tw_start = float(
                        env['client_TW'][next_client][0]
                    )

                    tw_end = float(
                        env['client_TW'][next_client][1]
                    )

                    waiting_time = max(
                        0.0,
                        tw_start - arrival_time
                    )

                    lateness = max(
                        0.0,
                        arrival_time - tw_end
                    )

                    service_start_time = (
                        arrival_time + waiting_time
                    )

                    actual_service_time = float(
                        env['client_service'][next_client]
                    )

                    completion_time = (
                        service_start_time
                        + actual_service_time
                    )

                    # --------------------------------------------------
                    # Original reward function
                    # --------------------------------------------------

                    reward = 1.0 / (dist + 1e-8)

                    # --------------------------------------------------
                    # Update time
                    # --------------------------------------------------

                    env['total_time'] = completion_time

                    env['current_route_time'] += (
                        travel_time
                        + waiting_time
                        + actual_service_time
                    )

                    # --------------------------------------------------
                    # Update capacity with actual demand
                    # --------------------------------------------------

                    env['Q'] -= actual_demand

                    env['Q'] = max(
                        0.0,
                        env['Q']
                    )

                    # --------------------------------------------------
                    # Update route and visited customers
                    # --------------------------------------------------

                    env['current_route'].append(
                        next_client
                    )

                    env['unvisited_clients'].remove(
                        next_client
                    )

                    env['current_state'] = current_state

                    # --------------------------------------------------
                    # Cooperation
                    # --------------------------------------------------

                    remaining_clients = list(
                        env['unvisited_clients']
                    )

                    for assigned_vehicle in dv_assignment[depot]:

                        other_env = envs[depot][
                            assigned_vehicle
                        ]

                        other_env['unvisited_clients'] = list(
                            remaining_clients
                        )

                    # --------------------------------------------------
                    # Next candidate states
                    # --------------------------------------------------

                    next_candidate_states = build_candidate_states(
                        env,
                        depot,
                        D,
                        remaining_clients
                    )

                    transition_done = (
                        len(remaining_clients) == 0
                    )

                    # --------------------------------------------------
                    # Store transition
                    # --------------------------------------------------

                    Agents.store_transition(
                        current_state,
                        reward,
                        next_candidate_states,
                        transition_done
                    )

                    env['next_state'] = current_state.copy()

                    # --------------------------------------------------
                    # Episode completion for this vehicle
                    # --------------------------------------------------

                    if not remaining_clients:

                        if env['current_route']:

                            env['total_routes'].append(
                                list(env['current_route'])
                            )

                            env['current_route'] = []

                        env['done'] = True

            # --------------------------------------------------
            # Learning update
            # --------------------------------------------------

            if (
                Q_update_frequency > 0
                and update_steps > 0
                and update_steps % Q_update_frequency == 0
            ):

                Agents.learn()

            # --------------------------------------------------
            # Target network update
            # --------------------------------------------------

            if (
                Target_update_freq > 0
                and update_steps > 0
                and update_steps % Target_update_freq == 0
            ):

                Agents.update_target_network()

            global_time_step += 0.1
            update_steps += 1

            # --------------------------------------------------
            # Completion check
            # --------------------------------------------------

            all_done = True

            for depot_id in envs.keys():

                for vehicle_id in envs[depot_id]:

                    if not envs[depot_id][vehicle_id]['done']:

                        all_done = False
                        break

                if not all_done:
                    break

        # --------------------------------------------------
        # Check episode completion
        # --------------------------------------------------

        if not all_done:

            unfinished = {}

            for depot_id in envs.keys():

                for vehicle_id in envs[depot_id]:

                    env = envs[depot_id][vehicle_id]

                    if not env['done']:

                        unfinished[(depot_id, vehicle_id)] = (
                            list(env['unvisited_clients'])
                        )

            raise RuntimeError(
                "Episode ended before all customers were served. "
                f"Episode: {episode}, "
                f"Unfinished vehicles/customers: {unfinished}. "
                "Check n_time_steps and route/time constraints."
            )

        # --------------------------------------------------
        # Store episode routes
        # --------------------------------------------------

        total_routes = {}

        for depot_id in envs.keys():

            total_routes[depot_id] = {}

            for vehicle_id in envs[depot_id]:

                total_routes[depot_id][vehicle_id] = (
                    copy.deepcopy(
                        envs[depot_id][vehicle_id]['total_routes']
                    )
                )

        # Keep the final episode scenario for VNS.
        Agents.last_scenario = copy.deepcopy(
            episode_scenario
        )

        Agents.last_episode_complete = True

        # Exploration probability is updated once per episode.
        if hasattr(
            Agents.Q_eval,
            'update_exploration_probability'
        ):

            Agents.Q_eval.update_exploration_probability()

    return Agents, total_routes