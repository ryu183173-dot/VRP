from PDQN_Network import *
from Reading_dataset import *
from scipy.spatial.distance import euclidean

import numpy as np
import copy
from tqdm import tqdm

from configurations import config

battery_capacity, vehicle_capacity, vehicle_velocity, \
    vehicle_energy_decay, energy_consumption_per_distance = config()


def energy_consumption(dist, current_energy, idle=False):
    energy = current_energy

    if idle:
        energy = current_energy * (1 - vehicle_energy_decay)
    else:
        energy -= dist * energy_consumption_per_distance

    return energy


# --------------------------------------------------
# Position helper
# --------------------------------------------------

def get_current_position(env, depot, D):
    current_id = env['current_state_id']

    if current_id in D:
        return (
            D[current_id][0],
            D[current_id][1]
        )

    return (
        env['client_coords'][current_id][0],
        env['client_coords'][current_id][1]
    )


# --------------------------------------------------
# Candidate-specific 9-dimensional state
# --------------------------------------------------

def build_candidate_states(env, depot, D, candidate_ids):
    """
    Create one 9-dimensional state for each candidate customer.

    State:
        0: current_x
        1: current_y
        2: target_x
        3: target_y
        4: remaining_energy
        5: remaining_capacity
        6: current_time
        7: target_time_window_start
        8: target_time_window_end
    """

    current_position = get_current_position(
        env,
        depot,
        D
    )

    candidate_states = []

    for client in candidate_ids:

        target_position = env['client_coords'][client]

        tw_start = env['client_TW'][client][0]
        tw_end = env['client_TW'][client][1]

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

    if len(candidate_states) == 0:
        return np.empty((0, 9), dtype=np.float32)

    return np.asarray(
        candidate_states,
        dtype=np.float32
    )


# --------------------------------------------------
# Main RL algorithm
# --------------------------------------------------

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
    beta=0.2
):

    # 9-dimensional candidate-specific state
    state_dim = 9

    # Kept for compatibility with the agent constructor.
    # The new network evaluates one candidate state at a time.
    action_dim = len(C) + len(D)

    envs = dict()

    # --------------------------------------------------
    # Cluster information
    # --------------------------------------------------

    cluster_demands = dict()
    cluster_TW = dict()
    cluster_coords = dict()
    cluster_service = dict()

    for depot in D.keys():

        cluster_demands[depot] = dict()
        cluster_TW[depot] = dict()
        cluster_coords[depot] = dict()
        cluster_service[depot] = dict()

        for client in Clusters[depot]:

            cluster_demands[depot][client] = Demands[client]
            cluster_TW[depot][client] = TW_C[client]
            cluster_coords[depot][client] = C[client]
            cluster_service[depot][client] = T_Service[client]

    # --------------------------------------------------
    # Double DQN Agent
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

    dv_assignment = dict()

    for v_num in range(n_vehicles):

        if v_num < len(D):

            if v_num not in dv_assignment.keys():
                dv_assignment[v_num] = []

            dv_assignment[v_num].append(v_num)

        else:
            dv_assignment[v_num - len(D)].append(v_num)

    print(dv_assignment)

    # --------------------------------------------------
    # Training episodes
    # --------------------------------------------------

    for episode in tqdm(
        list(range(n_episodes)),
        desc="PDQN Algorithm Episodes"
    ):

        cluster_demands = dict()
        cluster_TW = dict()
        cluster_coords = dict()

        for depot in D.keys():

            cluster_demands[depot] = dict()
            cluster_TW[depot] = dict()
            cluster_coords[depot] = dict()

            for client in Clusters[depot]:

                cluster_demands[depot][client] = Demands[client]
                cluster_TW[depot][client] = TW_C[client]
                cluster_coords[depot][client] = C[client]

        # --------------------------------------------------
        # Environment initialization
        # --------------------------------------------------

        for depot in D.keys():

            envs[depot] = dict()

            for v in dv_assignment[depot]:

                envs[depot][v] = {

                    'done': False,

                    'total_routes': [],
                    'current_route': [],

                    'total_time': 0,
                    'total_distance': 0,
                    'current_route_time': 0,

                    'Q': copy.deepcopy(Q),

                    'idle': False,

                    'client_demands': copy.deepcopy(
                        cluster_demands[depot]
                    ),

                    'client_TW': copy.deepcopy(
                        cluster_TW[depot]
                    ),

                    'client_coords': copy.deepcopy(
                        cluster_coords[depot]
                    ),

                    'client_service': copy.deepcopy(
                        cluster_service[depot]
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

                    'unvisited_clients': copy.deepcopy(
                        Clusters[depot]
                    ),

                    'energy': battery_capacity
                }

        global_time_step = 0
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

                for v in dv_assignment[depot]:

                    env = envs[depot][v]

                    # --------------------------------------------------
                    # Idle vehicle
                    # --------------------------------------------------

                    if (
                        global_time_step < env['total_time']
                        and env['idle']
                        and not env['done']
                    ):

                        reward = -1

                        env['energy'] = energy_consumption(
                            0,
                            env['energy'],
                            idle=True
                        )

                    # --------------------------------------------------
                    # Vehicle is ready to make a decision
                    # --------------------------------------------------

                    elif (
                        global_time_step >= env['total_time']
                        and not env['done']
                    ):

                        remaining_demands = set()

                        for rmd_item in env['unvisited_clients']:

                            remaining_demands.add(
                                env['client_demands'][rmd_item]
                            )

                        if 0 in remaining_demands:
                            remaining_demands.remove(0)

                        if len(remaining_demands):

                            min_demands = max(
                                0,
                                min(remaining_demands)
                            )

                            # --------------------------------------------------
                            # Return to depot
                            # --------------------------------------------------

                            if (
                                env['Q'] < min_demands
                                or env['current_route_time'] >= route_max_time
                                or env['energy'] <= 0
                            ):

                                next_client = action = depot
                                reward = 0

                                current_position = get_current_position(
                                    env,
                                    depot,
                                    D
                                )

                                state = np.array(
                                    [
                                        current_position[0],
                                        current_position[1],

                                        D[depot][0],
                                        D[depot][1],

                                        env['energy'],
                                        env['Q'],
                                        env['total_time'],

                                        0,
                                        0
                                    ],
                                    dtype=np.float32
                                )

                                if len(env['current_route']):

                                    env['total_routes'].append(
                                        env['current_route']
                                    )

                                    env['current_route'] = []

                                env['current_state'] = state
                                env['current_state_id'] = depot

                                env['Q'] = copy.deepcopy(Q)

                                env['current_route_time'] = 0

                                env['energy'] = battery_capacity

                            # --------------------------------------------------
                            # Select next customer
                            # --------------------------------------------------

                            else:

                                current_client = env['current_state_id']

                                env['idle'] = False

                                reward = 0

                                # SAP / feasible candidate filtering:
                                # retain customers whose demands fit
                                # the remaining vehicle capacity.
                                unseen_clients = []

                                for item in env['unvisited_clients']:

                                    if (
                                        env['Q']
                                        >= env['client_demands'][item]
                                    ):
                                        unseen_clients.append(item)

                                if len(unseen_clients):

                                    # Create one 9D state per candidate.
                                    candidate_states = build_candidate_states(
                                        env,
                                        depot,
                                        D,
                                        unseen_clients
                                    )

                                    # The agent returns the index of
                                    # the selected candidate.
                                    selected_index = Agents.choose_action(
                                        candidate_states
                                    )

                                    next_client = unseen_clients[
                                        selected_index
                                    ]

                                    # State corresponding to the
                                    # selected candidate.
                                    current_state = candidate_states[
                                        selected_index
                                    ].copy()

                                    # --------------------------------------------------
                                    # Time window examination
                                    # --------------------------------------------------

                                    tw_start = env['client_TW'][
                                        next_client
                                    ][0]

                                    tw_end = env['client_TW'][
                                        next_client
                                    ][1]

                                    current = get_current_position(
                                        env,
                                        depot,
                                        D
                                    )

                                    next_position = (
                                        env['client_coords'][next_client][0],
                                        env['client_coords'][next_client][1]
                                    )

                                    current_time = env['total_time']

                                    # --------------------------------------------------
                                    # Case 1: Within time window
                                    # --------------------------------------------------

                                    if (
                                        tw_start
                                        <= current_time
                                        <= tw_end
                                    ):

                                        env['idle'] = False

                                        action = next_client

                                        dist = euclidean(
                                            current,
                                            next_position
                                        )

                                        reward = 1 / dist

                                        env['total_time'] += dist

                                        env['total_time'] += (
                                            env['client_service'][next_client]
                                        )

                                        env['current_route_time'] += (
                                            dist / vehicle_velocity
                                        )

                                        env['current_route_time'] += (
                                            env['client_service'][next_client]
                                        )

                                        env['total_distance'] += dist

                                    # --------------------------------------------------
                                    # Case 2: Early arrival
                                    # --------------------------------------------------

                                    elif current_time < tw_start:

                                        env['idle'] = True

                                        time_difference = (
                                            tw_start - current_time
                                        )

                                        action = next_client

                                        env['current_route_time'] = tw_start

                                        dist = euclidean(
                                            current,
                                            next_position
                                        )

                                        reward = 1 / (
                                            dist
                                            + alpha * time_difference
                                        )

                                        env['total_time'] += dist

                                        env['total_time'] += (
                                            env['client_service'][next_client]
                                        )

                                        env['current_route_time'] += (
                                            dist / vehicle_velocity
                                        )

                                        env['current_route_time'] += (
                                            env['client_service'][next_client]
                                        )

                                        env['total_distance'] += dist

                                    # --------------------------------------------------
                                    # Case 3: Late arrival
                                    # --------------------------------------------------

                                    elif current_time > tw_end:

                                        env['idle'] = False

                                        time_difference = (
                                            current_time - tw_end
                                        )

                                        action = next_client

                                        dist = euclidean(
                                            current,
                                            next_position
                                        )

                                        reward = 1 / (
                                            dist
                                            + beta * time_difference
                                        )

                                        env['total_time'] += dist

                                        env['total_time'] += (
                                            env['client_service'][next_client]
                                        )

                                        env['current_route_time'] += (
                                            dist / vehicle_velocity
                                        )

                                        env['current_route_time'] += (
                                            env['client_service'][next_client]
                                        )

                                        env['total_distance'] += dist

                                    # --------------------------------------------------
                                    # Update vehicle energy
                                    # --------------------------------------------------

                                    env['energy'] = energy_consumption(
                                        dist,
                                        env['energy'],
                                        idle=False
                                    )

                                    # --------------------------------------------------
                                    # Update customer and capacity information
                                    # --------------------------------------------------

                                    env['current_state'] = current_state

                                    env['current_state_id'] = next_client

                                    env['current_route'].append(
                                        next_client
                                    )

                                    env['unvisited_clients'].remove(
                                        next_client
                                    )

                                    env['Q'] -= (
                                        env['client_demands'][next_client]
                                    )

                                    # --------------------------------------------------
                                    # Cooperation:
                                    # share remaining customers among
                                    # vehicles assigned to this depot.
                                    # --------------------------------------------------

                                    remained_clients = copy.deepcopy(
                                        env['unvisited_clients']
                                    )

                                    for assigned_vehicle in dv_assignment[depot]:

                                        envs[depot][
                                            assigned_vehicle
                                        ]['unvisited_clients'] = copy.deepcopy(
                                            remained_clients
                                        )

                                    # --------------------------------------------------
                                    # Build next candidate states
                                    # --------------------------------------------------

                                    next_candidates = []

                                    for client in remained_clients:

                                        if (
                                            env['Q']
                                            >= env['client_demands'][client]
                                        ):
                                            next_candidates.append(client)

                                    next_candidate_states = build_candidate_states(
                                        env,
                                        depot,
                                        D,
                                        next_candidates
                                    )

                                    # Episode terminal condition:
                                    # all customers in this depot cluster
                                    # have been assigned.
                                    transition_done = (
                                        len(remained_clients) == 0
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

                                    env['next_state'] = current_state

                        # --------------------------------------------------
                        # Learning update
                        # --------------------------------------------------

                        if (
                            update_steps % Q_update_frequency == 0
                            and update_steps > 0
                        ):

                            loss = Agents.learn()

                        # --------------------------------------------------
                        # Target network update
                        # --------------------------------------------------

                        if (
                            update_steps % Target_update_freq == 0
                            and update_steps > 0
                        ):

                            Agents.update_target_network()

                        # --------------------------------------------------
                        # Episode completion check
                        # --------------------------------------------------

                        if not len(env['unvisited_clients']):

                            env['done'] = True

                        else:

                            env['done'] = False

                        if env['done']:

                            Agents.Q_eval.update_exploration_probability()

                            env['total_routes'].append(
                                env['current_route']
                            )

                            env['current_route'] = []

            global_time_step += 0.1
            update_steps += 1

            # --------------------------------------------------
            # Check whether all vehicles are done
            # --------------------------------------------------

            all_done = True

            for d in envs.keys():

                for v in envs[d]:

                    if not envs[d][v]['done']:

                        all_done = False

        # --------------------------------------------------
        # Episode result
        # --------------------------------------------------

        total_dist = 0
        total_routes = dict()

        for e in envs.keys():

            total_routes[e] = dict()

            for v in envs[e]:

                total_routes[e][v] = []

                total_routes[e][v] = envs[e][v]['total_routes']

                total_dist += envs[e][v]['total_distance']

    return Agents, total_routes