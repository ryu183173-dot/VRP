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
# Constants
# ==================================================

STATE_DIM = 9
DEMAND_MIN_FACTOR = 0.8
DEMAND_MAX_FACTOR = 1.2
SERVICE_MIN_FACTOR = 0.8
SERVICE_MAX_FACTOR = 1.2
TRAVEL_MIN_FACTOR = 0.9
TRAVEL_MAX_FACTOR = 1.3
ENERGY_MIN_FACTOR = 0.9
ENERGY_MAX_FACTOR = 1.2

EPS = 1e-8


# ==================================================
# Energy consumption
# ==================================================

def energy_consumption(dist, current_energy, factor=1.0):
    """
    Energy consumption is proportional to distance.

    energy_used = distance * energy_consumption_per_distance
                  * dynamic_energy_factor
    """

    dist = max(0.0, float(dist))
    current_energy = float(current_energy)
    factor = float(factor)

    energy_used = (
        dist
        * float(energy_consumption_per_distance)
        * factor
    )

    return max(0.0, current_energy - energy_used)


def required_energy(dist, factor=ENERGY_MAX_FACTOR):
    """Conservative energy required for a given distance."""

    return (
        max(0.0, float(dist))
        * float(energy_consumption_per_distance)
        * float(factor)
    )


# ==================================================
# Position helper
# ==================================================

def get_current_position(env, depot, D):

    current_id = env["current_state_id"]

    if current_id in D:
        return tuple(D[current_id])

    if current_id not in env["client_coords"]:
        raise KeyError(
            f"Unknown current node: {current_id}"
        )

    return tuple(env["client_coords"][current_id])


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

        if client not in env["client_coords"]:
            raise KeyError(
                f"Customer {client} is not in this depot cluster."
            )

        target_position = env["client_coords"][client]

        tw_start, tw_end = env["client_TW"][client]

        state = np.array(
            [
                current_position[0],
                current_position[1],
                target_position[0],
                target_position[1],
                env["energy"],
                env["Q"],
                env["total_time"],
                tw_start,
                tw_end
            ],
            dtype=np.float32
        )

        candidate_states.append(state)

    if not candidate_states:
        return np.empty(
            (0, STATE_DIM),
            dtype=np.float32
        )

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
    Generate one reproducible scenario.

    Experimental assumptions:
        Demand factor:       U(0.8, 1.2)
        Service-time factor: U(0.8, 1.2)
        Travel-time factor:  U(0.9, 1.3)
        Energy factor:       U(0.9, 1.2)

    Actual demand and service time are stored in the
    scenario, but are not exposed to candidate filtering.
    """

    rng = np.random.default_rng(scenario_seed)

    actual_demands = {}
    actual_service = {}

    for client in C:

        base_demand = float(Demands[client])
        base_service = float(T_Service[client])

        actual_demands[client] = (
            base_demand
            * rng.uniform(
                DEMAND_MIN_FACTOR,
                DEMAND_MAX_FACTOR
            )
        )

        actual_service[client] = (
            base_service
            * rng.uniform(
                SERVICE_MIN_FACTOR,
                SERVICE_MAX_FACTOR
            )
        )

    infeasible_clients = [
        client
        for client, demand in actual_demands.items()
        if demand > float(max_capacity) + EPS
    ]

    if infeasible_clients:
        raise ValueError(
            "Dynamic scenario contains customers whose actual "
            "demand exceeds the full vehicle capacity. "
            f"Customers: {infeasible_clients}"
        )

    all_nodes = list(D.keys()) + list(C.keys())

    travel_factors = {}
    energy_factors = {}

    for from_node in all_nodes:

        for to_node in all_nodes:

            if from_node == to_node:
                continue

            travel_factors[(from_node, to_node)] = float(
                rng.uniform(
                    TRAVEL_MIN_FACTOR,
                    TRAVEL_MAX_FACTOR
                )
            )

            energy_factors[(from_node, to_node)] = float(
                rng.uniform(
                    ENERGY_MIN_FACTOR,
                    ENERGY_MAX_FACTOR
                )
            )

    return {
        "seed": int(scenario_seed),
        "actual_demands": actual_demands,
        "actual_service": actual_service,
        "travel_factors": travel_factors,
        "energy_factors": energy_factors
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

    travel_factor = float(
        scenario["travel_factors"].get(
            (from_id, to_id),
            1.0
        )
    )

    energy_factor = float(
        scenario["energy_factors"].get(
            (from_id, to_id),
            1.0
        )
    )

    base_travel_time = (
        dist / max(float(vehicle_velocity), EPS)
    )

    travel_time = base_travel_time * travel_factor

    return (
        dist,
        travel_time,
        energy_factor
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

    Capacity and battery are restored only after the
    vehicle reaches its depot.
    """

    if env["returning_to_depot"]:
        return

    current_position = get_current_position(
        env,
        depot,
        D
    )

    depot_position = tuple(D[depot])
    current_id = env["current_state_id"]

    dist, travel_time, energy_factor = (
        get_travel_information(
            current_id,
            depot,
            current_position,
            depot_position,
            scenario
        )
    )

    required = (
        dist
        * float(energy_consumption_per_distance)
        * energy_factor
    )

    if env["energy"] + EPS < required:
        raise RuntimeError(
            "Vehicle does not have enough energy to return "
            f"to depot {depot}. "
            f"Remaining energy={env['energy']:.6f}, "
            f"required energy={required:.6f}."
        )

    env["total_distance"] += dist

    env["energy"] = energy_consumption(
        dist,
        env["energy"],
        energy_factor
    )

    env["total_time"] += travel_time
    env["current_route_time"] += travel_time

    if env["current_route"]:

        env["total_routes"].append(
            list(env["current_route"])
        )

        env["current_route"] = []

    env["returning_to_depot"] = True


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

    Actual demand and actual service time are not used
    for candidate filtering.

    Conservative factors:
        Travel time: 1.3
        Energy:      1.2
        Service:     1.2
    """

    current_position = get_current_position(
        env,
        depot,
        D
    )

    depot_position = tuple(D[depot])

    feasible_candidates = []

    velocity = max(float(vehicle_velocity), EPS)

    for client in candidate_ids:

        target_position = tuple(
            env["client_coords"][client]
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

        # ------------------------------------------
        # Capacity feasibility using known demand
        # ------------------------------------------

        base_demand = float(Demands[client])

        if base_demand > env["Q"] + EPS:
            continue

        if base_demand > env["capacity"] + EPS:
            continue

        # ------------------------------------------
        # Conservative energy feasibility
        # ------------------------------------------

        conservative_energy = (
            required_energy(
                outbound_dist,
                ENERGY_MAX_FACTOR
            )
            + required_energy(
                return_dist,
                ENERGY_MAX_FACTOR
            )
        )

        # ------------------------------------------
        # Conservative travel-time feasibility
        # ------------------------------------------

        outbound_time = (
            outbound_dist / velocity
        ) * TRAVEL_MAX_FACTOR

        return_time = (
            return_dist / velocity
        ) * TRAVEL_MAX_FACTOR

        estimated_arrival = (
            env["total_time"] + outbound_time
        )

        tw_start = float(
            env["client_TW"][client][0]
        )

        estimated_waiting = max(
            0.0,
            tw_start - estimated_arrival
        )

        conservative_service_time = (
            float(T_Service[client])
            * SERVICE_MAX_FACTOR
        )

        # The paper's simplified battery model includes a constant
        # energy decay while the vehicle is idle (waiting/servicing).
        conservative_idle_energy = (
            float(vehicle_energy_decay)
            * (estimated_waiting + conservative_service_time)
        )
        if env["energy"] + EPS < conservative_energy + conservative_idle_energy:
            continue

        estimated_route_time = (
            env["current_route_time"]
            + outbound_time
            + estimated_waiting
            + conservative_service_time
            + return_time
        )

        if estimated_route_time > float(route_max_time) + EPS:
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

    state_dim = STATE_DIM
    action_dim = len(C) + len(D)

    # --------------------------------------------------
    # Input validation
    # --------------------------------------------------

    if not D:
        raise ValueError("At least one depot is required.")

    if n_vehicles < len(D):
        raise ValueError(
            "n_vehicles must be at least the number of depots."
        )

    if float(Q) <= 0:
        raise ValueError("Vehicle capacity must be positive.")

    if float(battery_capacity) <= 0:
        raise ValueError("Battery capacity must be positive.")

    if float(vehicle_velocity) <= 0:
        raise ValueError("Vehicle velocity must be positive.")

    if float(route_max_time) <= 0:
        raise ValueError("route_max_time must be positive.")

    # --------------------------------------------------
    # Cluster information
    # --------------------------------------------------

    cluster_TW = {}
    cluster_coords = {}

    for depot in D:

        if depot not in Clusters:
            raise KeyError(
                f"Cluster information is missing for depot {depot}."
            )

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

    # Preserve the current initial exploration setting.
    Agents.Q_eval.exploration_proba = 0.5

    # --------------------------------------------------
    # Vehicle assignment
    # --------------------------------------------------

    dv_assignment = {
        depot: []
        for depot in D
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

        # ----------------------------------------------
        # Reproducible dynamic scenario
        # ----------------------------------------------

        episode_scenario = generate_dynamic_scenario(
            C,
            D,
            Demands,
            T_Service,
            scenario_seed + episode,
            Q
        )

        envs = {}

        # ----------------------------------------------
        # Environment initialization
        # ----------------------------------------------

        for depot in D:

            envs[depot] = {}

            for vehicle_id in dv_assignment[depot]:

                envs[depot][vehicle_id] = {

                    "done": False,

                    "total_routes": [],
                    "current_route": [],

                    "total_time": 0.0,
                    "total_distance": 0.0,
                    "current_route_time": 0.0,

                    "Q": float(Q),
                    "capacity": float(Q),

                    "returning_to_depot": False,

                    # Actual demand/service are not passed
                    # to candidate feasibility filtering.
                    "client_demands": copy.deepcopy(
                        episode_scenario["actual_demands"]
                    ),

                    "client_TW": copy.deepcopy(
                        cluster_TW[depot]
                    ),

                    "client_coords": copy.deepcopy(
                        cluster_coords[depot]
                    ),

                    "client_service": copy.deepcopy(
                        episode_scenario["actual_service"]
                    ),

                    "current_state_id": depot,

                    "current_state": np.zeros(
                        state_dim,
                        dtype=np.float32
                    ),

                    "next_state": np.zeros(
                        state_dim,
                        dtype=np.float32
                    ),

                    "unvisited_clients": list(
                        Clusters[depot]
                    ),

                    "energy": float(battery_capacity)
                }

        global_time_step = 0.0
        update_steps = 0
        all_done = False

        # ----------------------------------------------
        # Main simulation loop
        # ----------------------------------------------

        while (
            not all_done
            and global_time_step <= n_time_steps
        ):

            for depot in D:

                for vehicle_id in dv_assignment[depot]:

                    env = envs[depot][vehicle_id]

                    if env["done"]:
                        continue

                    # Vehicle is travelling or serving.
                    if (
                        global_time_step + EPS
                        < env["total_time"]
                    ):
                        continue

                    # ----------------------------------
                    # Complete return trip
                    # ----------------------------------

                    if env["returning_to_depot"]:

                        env["current_state_id"] = depot

                        env["energy"] = float(
                            battery_capacity
                        )

                        env["Q"] = float(
                            env["capacity"]
                        )

                        env["current_route_time"] = 0.0

                        env["returning_to_depot"] = False

                    # ----------------------------------
                    # Completion check
                    # ----------------------------------

                    if not env["unvisited_clients"]:

                        if env["current_route"]:

                            env["total_routes"].append(
                                list(env["current_route"])
                            )

                            env["current_route"] = []

                        env["done"] = True
                        continue

                    # ----------------------------------
                    # SAP candidate list
                    # ----------------------------------

                    candidate_ids = list(
                        env["unvisited_clients"]
                    )

                    current_position = get_current_position(
                        env,
                        depot,
                        D
                    )

                    current_id = env["current_state_id"]

                    depot_position = tuple(D[depot])

                    return_dist = float(
                        euclidean(
                            current_position,
                            depot_position
                        )
                    )

                    # ----------------------------------
                    # Return decision
                    # ----------------------------------

                    base_demands_remaining = [
                        float(Demands[client])
                        for client in candidate_ids
                    ]

                    min_base_demand = (
                        min(base_demands_remaining)
                        if base_demands_remaining
                        else 0.0
                    )

                    return_energy_need = required_energy(
                        return_dist,
                        ENERGY_MAX_FACTOR
                    )

                    should_return = (

                        env["Q"] + EPS < min_base_demand

                        or env["current_route_time"]
                        >= route_max_time - EPS

                        or env["energy"] + EPS
                        < return_energy_need
                    )

                    if should_return:

                        if current_id == depot:
                            raise RuntimeError(
                                "Vehicle cannot serve any remaining "
                                f"customer from depot {depot}. "
                                "Check demand, capacity and constraints."
                            )

                        schedule_return_to_depot(
                            env,
                            depot,
                            D,
                            episode_scenario
                        )

                        continue

                    # ----------------------------------
                    # Feasible candidate filtering
                    # ----------------------------------

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

                        if current_id == depot:
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

                    # ----------------------------------
                    # Candidate states
                    # ----------------------------------

                    candidate_states = build_candidate_states(
                        env,
                        depot,
                        D,
                        feasible_candidates
                    )

                    selected_index = int(
                        Agents.choose_action(candidate_states)
                    )

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

                    # ----------------------------------
                    # Travel to customer
                    # ----------------------------------

                    next_position = tuple(
                        env["client_coords"][next_client]
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
                        env["total_time"]
                    )

                    arrival_time = (
                        departure_time + travel_time
                    )

                    # ----------------------------------
                    # Energy consumption
                    # ----------------------------------

                    energy_required = (
                        dist
                        * float(energy_consumption_per_distance)
                        * energy_factor
                    )

                    if env["energy"] + EPS < energy_required:
                        raise RuntimeError(
                            "Energy feasibility check failed "
                            "during travel to customer "
                            f"{next_client}."
                        )

                    env["energy"] = energy_consumption(
                        dist,
                        env["energy"],
                        energy_factor
                    )

                    env["total_distance"] += dist

                    env["current_state_id"] = next_client

                    # ----------------------------------
                    # Reveal actual demand at arrival
                    # ----------------------------------

                    actual_demand = float(
                        env["client_demands"][next_client]
                    )

                    # If actual demand exceeds remaining capacity,
                    # return to depot and retry after reloading.
                    if actual_demand > env["Q"] + EPS:

                        env["total_time"] = arrival_time

                        env["current_route_time"] += travel_time

                        schedule_return_to_depot(
                            env,
                            depot,
                            D,
                            episode_scenario
                        )

                        continue

                    # ----------------------------------
                    # Time-window evaluation
                    # ----------------------------------

                    tw_start = float(
                        env["client_TW"][next_client][0]
                    )

                    tw_end = float(
                        env["client_TW"][next_client][1]
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
                        env["client_service"][next_client]
                    )

                    completion_time = (
                        service_start_time
                        + actual_service_time
                    )

                    # Vehicle is stationary while waiting and receiving service.
                    idle_duration = waiting_time + actual_service_time
                    idle_energy_loss = float(vehicle_energy_decay) * idle_duration
                    if env["energy"] + EPS < idle_energy_loss:
                        raise RuntimeError(
                            f"Vehicle {vehicle_id} at depot {depot} exhausted its battery "
                            "during waiting/service idle time."
                        )
                    env["energy"] = max(0.0, env["energy"] - idle_energy_loss)

                    # ----------------------------------
                    # Reward
                    # ----------------------------------

                    # Paper equation (8): reward depends on arrival relative
                    # to the customer's time window. The on-time case is 1/d_ij.
                    if arrival_time < tw_start:
                        reward = 1.0 / (
                            dist + float(alpha) * (tw_start - arrival_time) + EPS
                        )
                    elif arrival_time > tw_end:
                        reward = 1.0 / (
                            dist + float(beta) * (arrival_time - tw_end) + EPS
                        )
                    else:
                        reward = 1.0 / (dist + EPS)

                    # ----------------------------------
                    # Update time
                    # ----------------------------------

                    env["total_time"] = completion_time

                    env["current_route_time"] += (
                        travel_time
                        + waiting_time
                        + actual_service_time
                    )

                    # ----------------------------------
                    # Update capacity
                    # ----------------------------------

                    env["Q"] -= actual_demand

                    env["Q"] = max(
                        0.0,
                        env["Q"]
                    )

                    # ----------------------------------
                    # Update route and visited customers
                    # ----------------------------------

                    env["current_route"].append(
                        next_client
                    )

                    env["unvisited_clients"].remove(
                        next_client
                    )

                    env["current_state"] = current_state

                    # ----------------------------------
                    # Cooperation
                    # ----------------------------------

                    remaining_clients = list(
                        env["unvisited_clients"]
                    )

                    for assigned_vehicle in dv_assignment[depot]:

                        other_env = envs[depot][
                            assigned_vehicle
                        ]

                        other_env["unvisited_clients"] = list(
                            remaining_clients
                        )

                    # ----------------------------------
                    # Next candidate states
                    # ----------------------------------

                    next_candidate_states = build_candidate_states(
                        env,
                        depot,
                        D,
                        remaining_clients
                    )

                    transition_done = (
                        len(remaining_clients) == 0
                    )

                    # ----------------------------------
                    # Store transition
                    # ----------------------------------

                    Agents.store_transition(
                        current_state,
                        reward,
                        next_candidate_states,
                        transition_done
                    )

                    env["next_state"] = current_state.copy()

                    # ----------------------------------
                    # Vehicle completion
                    # ----------------------------------

                    if not remaining_clients:

                        if env["current_route"]:

                            env["total_routes"].append(
                                list(env["current_route"])
                            )

                            env["current_route"] = []

                        env["done"] = True

            # ----------------------------------------------
            # Learning update
            # ----------------------------------------------

            if (
                Q_update_frequency > 0
                and update_steps > 0
                and update_steps % Q_update_frequency == 0
            ):

                Agents.learn()

            # ----------------------------------------------
            # Target network update
            # ----------------------------------------------

            if (
                Target_update_freq > 0
                and update_steps > 0
                and update_steps % Target_update_freq == 0
            ):

                Agents.update_target_network()

            global_time_step += 0.1
            update_steps += 1

            # ----------------------------------------------
            # Completion check
            # ----------------------------------------------

            all_done = all(
                envs[depot_id][vehicle_id]["done"]
                for depot_id in envs
                for vehicle_id in envs[depot_id]
            )

        # ----------------------------------------------
        # Check episode completion
        # ----------------------------------------------

        if not all_done:

            unfinished = {}

            for depot_id in envs:

                for vehicle_id in envs[depot_id]:

                    env = envs[depot_id][vehicle_id]

                    if not env["done"]:

                        unfinished[(depot_id, vehicle_id)] = (
                            list(env["unvisited_clients"])
                        )

            raise RuntimeError(
                "Episode ended before all customers were served. "
                f"Episode: {episode}, "
                f"Unfinished vehicles/customers: {unfinished}. "
                "Check n_time_steps and route/time constraints."
            )

        # ----------------------------------------------
        # Store episode routes
        # ----------------------------------------------

        total_routes = {}

        for depot_id in envs:

            total_routes[depot_id] = {}

            for vehicle_id in envs[depot_id]:

                total_routes[depot_id][vehicle_id] = (
                    copy.deepcopy(
                        envs[depot_id][vehicle_id]["total_routes"]
                    )
                )

        # ----------------------------------------------
        # Store final scenario for VNS
        # ----------------------------------------------

        Agents.last_scenario = copy.deepcopy(
            episode_scenario
        )

        Agents.last_episode_complete = True

        # ----------------------------------------------
        # Exploration probability update
        # ----------------------------------------------

        Agents.Q_eval.update_exploration_probability()

    return Agents, total_routes