import copy
import numpy as np
from scipy.spatial.distance import euclidean


def route_decompose(route):
    """Split a depot-delimited route into customer-only trips."""
    if not route:
        return []
    depot = route[0]
    routes, current = [], []
    for node in route[1:]:
        if node == depot:
            if current:
                routes.append(current)
                current = []
        else:
            current.append(node)
    if current:
        routes.append(current)
    return routes


def generating_clusters(coordinates, depot_coords):
    """
    Depot clustering following the proximity-ratio/capacity logic of
    Shahbazian et al. (2025), Algorithm 4. Customers with a smaller
    nearest/second-nearest distance ratio are assigned first. A balanced
    target is used; if both preferred clusters are full, the nearest depot
    receives the remaining customer so every customer is assigned exactly once.
    """
    customer_ids = list(coordinates.keys())
    depot_ids = list(depot_coords.keys())
    if not depot_ids:
        raise ValueError("At least one depot is required.")
    clusters = {d: [] for d in depot_ids}
    if not customer_ids:
        return clusters

    target = max(1, int(len(customer_ids) / len(depot_ids)))
    ranked = []
    for c in customer_ids:
        distances = sorted(
            ((float(euclidean(coordinates[c], depot_coords[d])), d) for d in depot_ids),
            key=lambda x: (x[0], depot_ids.index(x[1]))
        )
        d1_dist, d1 = distances[0]
        if len(distances) == 1:
            d2_dist, d2 = float('inf'), d1
            ratio = 0.0
        else:
            d2_dist, d2 = distances[1]
            ratio = d1_dist / max(d2_dist, 1e-12)
        ranked.append((ratio, customer_ids.index(c), c, d1, d2))

    ranked.sort(key=lambda x: (x[0], x[1]))
    for _, _, c, d1, d2 in ranked:
        if len(clusters[d1]) < target:
            chosen = d1
        elif d2 != d1 and len(clusters[d2]) < target:
            chosen = d2
        else:
            # Algorithm 4's fallback assigns to the closest depot d1.
            chosen = d1
        clusters[chosen].append(c)

    # Preserve the original customer order within each cluster.
    order = {c: i for i, c in enumerate(customer_ids)}
    for d in depot_ids:
        clusters[d].sort(key=order.__getitem__)
    assigned = [c for values in clusters.values() for c in values]
    if len(assigned) != len(customer_ids) or len(set(assigned)) != len(customer_ids):
        raise RuntimeError("Depot clustering failed to assign each customer exactly once.")
    return clusters