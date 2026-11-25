from scipy.spatial import distance
import numpy as np
import random
import time
import copy


def _is_time_up(start_time, time_limit_sec):
    if time_limit_sec is None:
        return False
    return (time.time() - start_time) >= time_limit_sec


def _validate_positive_finite(val, name):
    if val is None or isinstance(val, bool):
        raise ValueError(f"Invalid {name}: {val}")
    fval = float(val)
    if not np.isfinite(fval) or fval <= 0.0:
        raise ValueError(f"Non-positive or non-finite {name}: {val}")
    return fval


def _validate_nonneg_finite(val, name):
    if val is None or isinstance(val, bool):
        raise ValueError(f"Invalid {name}: {val}")
    fval = float(val)
    if not np.isfinite(fval) or fval < 0.0:
        raise ValueError(f"Negative or non-finite {name}: {val}")
    return fval


def _get_demand(demands, client, max_cap):
    if demands is None or client not in demands:
        raise KeyError(f"Missing demand for node: {client}")
    d = _validate_nonneg_finite(demands[client], f"demand[{client}]")
    if d > max_cap + 1e-12:
        raise ValueError(
            f"Single-customer demand exceeds vehicle capacity: client={client}, "
            f"demand={d}, max_cap={max_cap}"
        )
    return d


def _get_distance(dist_matrix, u, v):
    if dist_matrix is None or (u, v) not in dist_matrix:
        raise KeyError(f"Missing distance for edge: ({u}, {v})")
    return _validate_nonneg_finite(dist_matrix[(u, v)], f"distance[({u}, {v})]")


def compute_distances(clients):
    distances = {}
    for from_client in clients.keys():
        for to_client in clients.keys():
            d = float(distance.euclidean(clients[from_client], clients[to_client]))
            distances[(from_client, to_client)] = _validate_nonneg_finite(
                d, f"euclidean_distance[({from_client}, {to_client})]"
            )
    return distances


def tour_distance(tour, dist_matrix):
    if len(tour) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(tour)):
        total += _get_distance(dist_matrix, tour[i - 1], tour[i])
    return float(total)


def two_opt(route, i, j):
    """先頭・末尾のデポを保持したまま区間 [i, j] を反転する。"""
    if len(route) < 3 or i < 1 or j >= len(route) - 1 or i > j:
        return route[:]
    return route[:i] + route[i:j + 1][::-1] + route[j + 1:]


def subroute_time_eval(
    subroute,
    distance_matrix,
    C_TW,
    C_service,
    alpha=0.1,
    beta=0.2,
    velocity=1.0,
):
    """式(10)に基づき、サブルート [depot, c1, ..., ck, depot] の時間枠評価値を返す。"""
    vel = _validate_positive_finite(velocity, "velocity")
    alpha_v = _validate_nonneg_finite(alpha, "alpha")
    beta_v = _validate_nonneg_finite(beta, "beta")

    if len(subroute) < 3:
        return 0.0

    tw_penalty = 0.0
    curr_time = 0.0
    for idx in range(len(subroute) - 1):
        u = subroute[idx]
        v = subroute[idx + 1]
        travel_time = _get_distance(distance_matrix, u, v) / vel
        curr_time += travel_time

        if C_TW is not None and v in C_TW:
            ready = _validate_nonneg_finite(C_TW[v][0], f"TW[{v}][0]")
            due = _validate_nonneg_finite(C_TW[v][1], f"TW[{v}][1]")
            if curr_time < ready:
                tw_penalty += alpha_v * (ready - curr_time)
                curr_time = ready
            elif curr_time > due:
                tw_penalty += beta_v * (curr_time - due)

        if C_service is not None and v in C_service:
            svc = _validate_nonneg_finite(C_service[v], f"service_time[{v}]")
            curr_time += svc

    return float(tw_penalty)


def evaluate_subroute_cost(
    subroute,
    dist_matrix,
    TW,
    ST,
    alpha=0.1,
    beta=0.2,
    gamma=1.0,
    delta=1.0,
    velocity=1.0,
):
    """式(9)に従い、1サブルートの Cost = gamma * Distance + delta * Time Evaluation を返す。"""
    gamma_v = _validate_nonneg_finite(gamma, "gamma")
    delta_v = _validate_nonneg_finite(delta, "delta")
    dist = tour_distance(subroute, dist_matrix)
    time_eval = subroute_time_eval(
        subroute,
        dist_matrix,
        TW,
        ST,
        alpha=alpha,
        beta=beta,
        velocity=velocity,
    )
    cost = gamma_v * dist + delta_v * time_eval
    return {
        "distance": float(dist),
        "time_evaluation": float(time_eval),
        "cost": float(cost),
    }


def local_search(
    subroute,
    dist_matrix,
    TW=None,
    ST=None,
    alpha=0.1,
    beta=0.2,
    gamma=1.0,
    delta=1.0,
    velocity=1.0,
    operator=two_opt,
    start_time=None,
    time_limit_sec=None,
):
    """サブルート [depot, ..., depot] 内で式(9)のコストに基づき Two-Opt 改善を行う。"""
    if len(subroute) < 3:
        return subroute[:]

    best = subroute[:]
    best_cost = evaluate_subroute_cost(
        best,
        dist_matrix,
        TW,
        ST,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        velocity=velocity,
    )["cost"]

    improved = True
    while improved:
        if start_time is not None and _is_time_up(start_time, time_limit_sec):
            break
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i, len(best) - 1):
                if start_time is not None and _is_time_up(start_time, time_limit_sec):
                    return best
                cand = operator(best, i, j)
                cand_cost = evaluate_subroute_cost(
                    cand,
                    dist_matrix,
                    TW,
                    ST,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    delta=delta,
                    velocity=velocity,
                )["cost"]
                if cand_cost < best_cost - 1e-12:
                    best = cand
                    best_cost = cand_cost
                    improved = True
                    break
            if improved:
                break
    return best


def shaking(subroute, k, operator=two_opt):
    """サブルートに k 回の Two-Opt 摂動を加える。"""
    curr = subroute[:]
    if len(curr) < 3:
        return curr
    num_clients = len(curr) - 2
    for _ in range(k):
        if num_clients >= 2:
            i, j = sorted(random.sample(range(1, len(curr) - 1), 2))
        else:
            i = j = 1
        curr = operator(curr, i, j)
    return curr


def split_into_subroutes(solution, depot):
    """ルート列を [depot, ..., depot] のサブルート群に分解する。"""
    if not solution:
        return []
    subroutes = []
    curr = []
    for node in solution:
        if node == depot:
            if len(curr) > 1:
                curr.append(depot)
                if len(curr) > 2:
                    subroutes.append(curr)
            curr = [depot]
        else:
            if not curr:
                curr = [depot]
            curr.append(node)
    if len(curr) > 1 and curr[-1] != depot:
        curr.append(depot)
        if len(curr) > 2:
            subroutes.append(curr)
    return subroutes


def flatten_subroutes(subroutes, depot):
    """サブルート群を1本のルート列へ戻す。"""
    if not subroutes:
        return [depot, depot]
    flat = []
    for sr in subroutes:
        if len(sr) > 2:
            if flat and flat[-1] == depot and sr[0] == depot:
                flat.extend(sr[1:])
            else:
                flat.extend(sr)
    return flat if flat else [depot, depot]


def repair_capacity_subroutes(subroutes, demands, max_cap, depot):
    """需要量が容量制約を超える場合、デポ帰還でサブルートを分割する（単独超過・欠損は検出）。"""
    cap = _validate_positive_finite(max_cap, "max_cap")
    repaired = []
    for sr in subroutes:
        clients = [n for n in sr if n != depot]
        if not clients:
            continue
        curr_clients = []
        curr_load = 0.0
        for c in clients:
            d = _get_demand(demands, c, cap)
            if curr_clients and (curr_load + d > cap + 1e-12):
                repaired.append([depot] + curr_clients + [depot])
                curr_clients = [c]
                curr_load = d
            else:
                curr_clients.append(c)
                curr_load += d
        if curr_clients:
            repaired.append([depot] + curr_clients + [depot])
    return repaired


def evaluate_solution_breakdown(
    solution,
    depot,
    demands,
    max_cap,
    dist_matrix,
    TW,
    ST,
    alpha=0.1,
    beta=0.2,
    gamma=1.0,
    delta=1.0,
    dist_rate=None,
    time_rate=None,
    velocity=1.0,
):
    """距離・時間枠評価・合計コスト・サブルート数を返す。"""
    g = _validate_nonneg_finite(dist_rate if dist_rate is not None else gamma, "gamma/dist_rate")
    d_rate = _validate_nonneg_finite(time_rate if time_rate is not None else delta, "delta/time_rate")
    subroutes = split_into_subroutes(solution, depot)
    total_dist = sum(tour_distance(sr, dist_matrix) for sr in subroutes)
    total_time_eval = sum(
        subroute_time_eval(
            sr,
            dist_matrix,
            TW,
            ST,
            alpha=alpha,
            beta=beta,
            velocity=velocity,
        )
        for sr in subroutes
    )
    total_cost = g * total_dist + d_rate * total_time_eval
    return {
        "distance": float(total_dist),
        "time_evaluation": float(total_time_eval),
        "cost": float(total_cost),
        "num_subroutes": len(subroutes),
    }


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
    gamma=1.0,
    delta=1.0,
    dist_rate=None,
    time_rate=None,
    velocity=1.0,
    k_max=50,
    operator=two_opt,
    time_limit_sec=900,
    seed=None,
    verbose=True,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    g = _validate_nonneg_finite(dist_rate if dist_rate is not None else gamma, "gamma/dist_rate")
    d_rate = _validate_nonneg_finite(time_rate if time_rate is not None else delta, "delta/time_rate")
    _validate_positive_finite(velocity, "velocity")

    start_time = time.time()
    total_exploration_time = 0.0
    total_exploitation_time = 0.0

    # 1. 初期解（容量修復前）の評価
    raw_subroutes = split_into_subroutes(tour, depot)
    raw_tour = flatten_subroutes(raw_subroutes, depot)
    raw_stats = evaluate_solution_breakdown(
        raw_tour, depot, demands, max_cap, dist_matrix, TW, ST,
        alpha=alpha, beta=beta, gamma=g, delta=d_rate, velocity=velocity
    )

    # 2. 容量修復後の初期解の評価
    init_subroutes = repair_capacity_subroutes(raw_subroutes, demands, max_cap, depot)
    best_subroutes = copy.deepcopy(init_subroutes)
    best_tour = flatten_subroutes(best_subroutes, depot)
    init_stats = evaluate_solution_breakdown(
        best_tour, depot, demands, max_cap, dist_matrix, TW, ST,
        alpha=alpha, beta=beta, gamma=g, delta=d_rate, velocity=velocity
    )
    best_cost = init_stats["cost"]

    if verbose:
        print(
            f"[VNS RAW INIT  | depot={depot}] dist={raw_stats['distance']:.4f}, "
            f"time_eval={raw_stats['time_evaluation']:.4f}, cost={raw_stats['cost']:.4f}, "
            f"subroutes={raw_stats['num_subroutes']}"
        )
        print(
            f"[VNS REPAIRED  | depot={depot}] dist={init_stats['distance']:.4f}, "
            f"time_eval={init_stats['time_evaluation']:.4f}, cost={init_stats['cost']:.4f}, "
            f"subroutes={init_stats['num_subroutes']}"
        )

    k = 1
    while k <= k_max and not _is_time_up(start_time, time_limit_sec):
        cand_subroutes = []
        for sr in best_subroutes:
            if _is_time_up(start_time, time_limit_sec):
                break

            t0 = time.time()
            shaken = shaking(sr, k, operator=operator)
            total_exploration_time += time.time() - t0

            if _is_time_up(start_time, time_limit_sec):
                break

            t1 = time.time()
            improved_sr = local_search(
                shaken,
                dist_matrix,
                TW=TW,
                ST=ST,
                alpha=alpha,
                beta=beta,
                gamma=g,
                delta=d_rate,
                velocity=velocity,
                operator=operator,
                start_time=start_time,
                time_limit_sec=time_limit_sec,
            )
            total_exploitation_time += time.time() - t1

            cand_subroutes.append(improved_sr)

        if _is_time_up(start_time, time_limit_sec):
            break

        cand_subroutes = repair_capacity_subroutes(cand_subroutes, demands, max_cap, depot)
        cand_tour = flatten_subroutes(cand_subroutes, depot)
        cand_stats = evaluate_solution_breakdown(
            cand_tour, depot, demands, max_cap, dist_matrix, TW, ST,
            alpha=alpha, beta=beta, gamma=g, delta=d_rate, velocity=velocity
        )

        if cand_stats["cost"] < best_cost - 1e-12:
            best_subroutes = copy.deepcopy(cand_subroutes)
            best_tour = cand_tour
            best_cost = cand_stats["cost"]
            if verbose:
                print(
                    f"[VNS IMPROVE | depot={depot}, k={k}] dist={cand_stats['distance']:.4f}, "
                    f"time_eval={cand_stats['time_evaluation']:.4f}, cost={cand_stats['cost']:.4f}, "
                    f"subroutes={cand_stats['num_subroutes']}"
                )
            k = 1
        else:
            k += 1

    # 3. VNS後の最終解の評価
    final_stats = evaluate_solution_breakdown(
        best_tour, depot, demands, max_cap, dist_matrix, TW, ST,
        alpha=alpha, beta=beta, gamma=g, delta=d_rate, velocity=velocity
    )
    if verbose:
        print(
            f"[VNS FINAL     | depot={depot}] dist={final_stats['distance']:.4f}, "
            f"time_eval={final_stats['time_evaluation']:.4f}, cost={final_stats['cost']:.4f}, "
            f"subroutes={final_stats['num_subroutes']}"
        )

    return best_tour, total_exploration_time, total_exploitation_time