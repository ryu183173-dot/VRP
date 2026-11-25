from scipy.spatial import distance
import numpy as np
import random
import time
import copy


def compute_distances(clients):
    distances = {}
    for from_client in clients.keys():
        for to_client in clients.keys():
            distances[(from_client, to_client)] = float(
                distance.euclidean(clients[from_client], clients[to_client])
            )
    return distances


def tour_distance(tour, dist_matrix):
    if len(tour) < 2:
        return 0.0
    return float(
        sum(dist_matrix[(tour[i - 1], tour[i])] for i in range(1, len(tour)))
    )


def two_opt(route, i, j):
    """先頭・末尾のデポを保持したまま区間 [i, j] を反転する"""
    if len(route) < 4 or i < 1 or j >= len(route) - 1 or i >= j:
        return route[:]
    return route[:i] + route[i:j + 1][::-1] + route[j + 1:]


def local_search(subroute, dist_matrix, operator=two_opt):
    """サブルート [depot, ..., depot] 内の距離を Two-Opt で改善する。"""
    if len(subroute) < 4:
        return subroute[:]
    best = subroute[:]
    best_dist = tour_distance(best, dist_matrix)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                if j - i == 1:
                    continue
                cand = operator(best, i, j)
                cand_dist = tour_distance(cand, dist_matrix)
                if cand_dist < best_dist - 1e-12:
                    best = cand
                    best_dist = cand_dist
                    improved = True
                    break
            if improved:
                break
    return best


def shaking(subroute, k, operator=two_opt):
    """サブルートに k 回の摂動を加える。"""
    curr = subroute[:]
    if len(curr) < 4:
        return curr
    for _ in range(k):
        i, j = sorted(random.sample(range(1, len(curr) - 1), 2))
        if j - i >= 1:
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
    """サブルートの需要合計が max_cap を超えないようデポ帰還で分割する。"""
    repaired = []
    for sr in subroutes:
        clients = [n for n in sr if n != depot]
        if not clients:
            continue
        curr_clients = []
        curr_load = 0.0
        for c in clients:
            d = float(demands.get(c, 0.0))
            if curr_clients and (curr_load + d > max_cap + 1e-12):
                repaired.append([depot] + curr_clients + [depot])
                curr_clients = [c]
                curr_load = d
            else:
                curr_clients.append(c)
                curr_load += d
        if curr_clients:
            repaired.append([depot] + curr_clients + [depot])
    return repaired


def subroute_time_eval(subroute, distance_matrix, C_TW, C_service, alpha=0.1, beta=0.2, velocity=1.0):
    """サブルート [depot, c1, ..., ck, depot] の時間枠早期・遅延ペナルティを計算する。"""
    if len(subroute) < 3:
        return 0.0
    penalty = 0.0
    curr_time = 0.0
    vel = max(float(velocity), 1e-12)
    for idx in range(len(subroute) - 1):
        u = subroute[idx]
        v = subroute[idx + 1]
        travel_time = distance_matrix.get((u, v), 0.0) / vel
        curr_time += travel_time
        if v in C_TW:
            ready, due = C_TW[v][0], C_TW[v][1]
            if curr_time < ready:
                penalty += alpha * (ready - curr_time)
                curr_time = ready
            elif curr_time > due:
                penalty += beta * (curr_time - due)
            curr_time += float(C_service.get(v, 0.0))
    return float(penalty)


def evaluate_solution_breakdown(solution, depot, demands, max_cap, dist_matrix, TW, ST,
                                alpha=0.1, beta=0.2, dist_rate=1.0, time_rate=1.0, velocity=1.0):
    """距離・時間枠ペナルティ・合計コスト・サブルート数を返す。"""
    subroutes = split_into_subroutes(solution, depot)
    total_dist = sum(tour_distance(sr, dist_matrix) for sr in subroutes)
    total_time_pen = sum(
        subroute_time_eval(sr, dist_matrix, TW, ST, alpha=alpha, beta=beta, velocity=velocity)
        for sr in subroutes
    )
    total_cost = dist_rate * total_dist + time_rate * total_time_pen
    return {
        "distance": float(total_dist),
        "time_penalty": float(total_time_pen),
        "cost": float(total_cost),
        "num_subroutes": len(subroutes),
    }


def vns(tour, depot, demands, max_cap, dist_matrix, TW, ST,
        alpha=0.1, beta=0.2, dist_rate=1.0, time_rate=1.0, velocity=1.0,
        k_max=50, operator=two_opt, time_limit_sec=900, verbose=True):
    start_time = time.time()
    total_exploration_time = 0.0
    total_exploitation_time = 0.0

    init_subroutes = repair_capacity_subroutes(
        split_into_subroutes(tour, depot), demands, max_cap, depot
    )
    best_subroutes = copy.deepcopy(init_subroutes)
    best_tour = flatten_subroutes(best_subroutes, depot)

    init_stats = evaluate_solution_breakdown(
        best_tour, depot, demands, max_cap, dist_matrix, TW, ST,
        alpha=alpha, beta=beta, dist_rate=dist_rate, time_rate=time_rate, velocity=velocity
    )
    best_cost = init_stats["cost"]

    if verbose:
        print(
            f"[VNS INIT  | depot={depot}] dist={init_stats['distance']:.4f}, "
            f"pen={init_stats['time_penalty']:.4f}, cost={init_stats['cost']:.4f}, "
            f"subroutes={init_stats['num_subroutes']}"
        )

    k = 1
    while k <= k_max and (time.time() - start_time) < time_limit_sec:
        cand_subroutes = []
        for sr in best_subroutes:
            t0 = time.time()
            shaken = shaking(sr, k, operator=operator)
            total_exploration_time += time.time() - t0

            t1 = time.time()
            improved_sr = local_search(shaken, dist_matrix, operator=operator)
            total_exploitation_time += time.time() - t1

            cand_subroutes.append(improved_sr)

        cand_subroutes = repair_capacity_subroutes(cand_subroutes, demands, max_cap, depot)
        cand_tour = flatten_subroutes(cand_subroutes, depot)
        cand_stats = evaluate_solution_breakdown(
            cand_tour, depot, demands, max_cap, dist_matrix, TW, ST,
            alpha=alpha, beta=beta, dist_rate=dist_rate, time_rate=time_rate, velocity=velocity
        )

        if cand_stats["cost"] < best_cost - 1e-12:
            best_subroutes = copy.deepcopy(cand_subroutes)
            best_tour = cand_tour
            best_cost = cand_stats["cost"]
            if verbose:
                print(
                    f"[VNS IMPROVE | depot={depot}, k={k}] dist={cand_stats['distance']:.4f}, "
                    f"pen={cand_stats['time_penalty']:.4f}, cost={cand_stats['cost']:.4f}, "
                    f"subroutes={cand_stats['num_subroutes']}"
                )
            k = 1
        else:
            k += 1

    final_stats = evaluate_solution_breakdown(
        best_tour, depot, demands, max_cap, dist_matrix, TW, ST,
        alpha=alpha, beta=beta, dist_rate=dist_rate, time_rate=time_rate, velocity=velocity
    )
    if verbose:
        print(
            f"[VNS FINAL | depot={depot}] dist={final_stats['distance']:.4f}, "
            f"pen={final_stats['time_penalty']:.4f}, cost={final_stats['cost']:.4f}, "
            f"subroutes={final_stats['num_subroutes']}"
        )

    return best_tour, total_exploration_time, total_exploitation_time