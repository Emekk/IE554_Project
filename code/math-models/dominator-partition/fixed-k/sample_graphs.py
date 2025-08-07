import itertools
import numpy as np


# Closed neighborhoods function
def closed_neighborhoods(V, E):
    neighborhoods = {v: set() for v in V}
    for u, v in E:
        neighborhoods[u].add(v)
        neighborhoods[v].add(u)
    for v in V:
        neighborhoods[v].add(v)  # Add self to closed neighborhood

    return neighborhoods

# Distance-2 graph function
def distance2_graph(V, E):
    # Build adjacency list
    adj = adjacency_list(V, E)

    # Construct edges of the square graph
    E2 = set()
    for u in V:
        # Distance-1 neighbors
        reachable = set(adj[u])
        # Add distance-2 neighbors
        for w in adj[u]:
            reachable |= adj[w]
        # Remove self if present
        reachable.discard(u)
        # Add edge for each reachable v
        for v in reachable:
            E2.add(frozenset({u, v}))

    return V, E2

def build_complement_graph(V, E):
    complement_edges = set()
    for u in V:
        for v in V:
            if u != v and frozenset({u, v}) not in E:
                complement_edges.add(frozenset({u, v}))
    return V, complement_edges

def adjacency_list(V, E):
    adj_list = {v: set() for v in V}
    for u, v in E:
        adj_list[u].add(v)
        adj_list[v].add(u)
    return adj_list

def bron_kerbosch_pivot(R, P, X, adj):
    if not P and not X and len(R) > 1:
        yield R.copy()
        return
    u = max(P.union(X), key=lambda v: len(adj[v]))
    for v in list(P - adj[u]):
        R_v = R.union({v})
        P_v = P.intersection(adj[v])
        X_v = X.intersection(adj[v])
        yield from bron_kerbosch_pivot(R_v, P_v, X_v, adj)
        P.remove(v)
        X.add(v)

def get_maximal_independent_sets(V, E):
    V, E = distance2_graph(V, E)
    V, E = build_complement_graph(V, E)
    adj = adjacency_list(V, E)
    return bron_kerbosch_pivot(R=set(), P=set(V), X=set(), adj=adj)

def build_random_graph(N, p, connected=True, seed=0):
    """
    Generate a random graph with N vertices and edge probability p.
    Returns a tuple (V, E) where V is the set of vertices and E is the set of edges.
    """
    np.random.seed(seed)
    V = set(range(1, N + 1))
    E = set()
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            if np.random.rand() < p:
                E.add(frozenset({u, v}))
    
    # ensure the graph is a connected graph if required
    if connected:
        # Union-Find (Disjoint Set) implementation
        parent = {v: v for v in V}

        def find(v):
            while parent[v] != v:
                parent[v] = parent[parent[v]]  # Path compression
                v = parent[v]
            return v

        def union(u, v):
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                parent[root_v] = root_u

        # Initially union all existing edges
        for edge in E:
            u, v = tuple(edge)
            union(u, v)

        # Connect disconnected components
        components = {}
        for v in V:
            root = find(v)
            components.setdefault(root, []).append(v)

        component_roots = list(components.keys())
        for i in range(len(component_roots) - 1):
            u = components[component_roots[i]][0]
            v = components[component_roots[i + 1]][0]
            E.add(frozenset({u, v}))
            union(u, v)  # Update the structure as we connect components

    return V, E


# SAMPLE GRAPHS
tree3 = (
    {1, 2, 3},
    {
        frozenset({1, 2}),
        frozenset({1, 3}),
    },
)
tree4_path = (
    {1, 2, 3, 4},
    {
        frozenset({1, 2}),
        frozenset({2, 3}),
        frozenset({3, 4}),
    },
)
tree4_star = (
    {1, 2, 3, 4},
    {
        frozenset({1, 2}),
        frozenset({1, 3}),
        frozenset({1, 4}),
    },
)
tree5_path = (
    {1, 2, 3, 4, 5},
    {
        frozenset({1, 2}),
        frozenset({2, 3}),
        frozenset({3, 4}),
        frozenset({4, 5}),
    },
)
tree5_star = (
    {1, 2, 3, 4, 5},
    {
        frozenset({1, 2}),
        frozenset({1, 3}),
        frozenset({1, 4}),
        frozenset({1, 5}),
    },
)
tree5_fork = (
    {1, 2, 3, 4, 5},
    {
        frozenset({1, 2}),
        frozenset({2, 3}),
        frozenset({3, 4}),
        frozenset({3, 5}),
    },
)
tree6_star = (
    {1, 2, 3, 4, 5, 6},
    {
        frozenset({1, 2}),
        frozenset({1, 3}),
        frozenset({1, 4}),
        frozenset({1, 5}),
        frozenset({1, 6}),
    },
)
tree6_path = (
    {1, 2, 3, 4, 5, 6},
    {
        frozenset({1, 2}),
        frozenset({2, 3}),
        frozenset({3, 4}),
        frozenset({4, 5}),
        frozenset({5, 6}),
    },
)
tree6_fork = (
    {1, 2, 3, 4, 5, 6},
    {
        frozenset({1, 2}),
        frozenset({2, 3}),
        frozenset({3, 4}),
        frozenset({3, 5}),
        frozenset({3, 6}),
    },
)
tree6_doublefork = (
    {1, 2, 3, 4, 5, 6},
    {
        frozenset({1, 3}),
        frozenset({2, 3}),
        frozenset({3, 4}),
        frozenset({4, 5}),
        frozenset({4, 6}),
    },
)
tree6_lollipop = (
    {1, 2, 3, 4, 5, 6},
    {
        frozenset({1, 2}),
        frozenset({2, 3}),
        frozenset({3, 4}),
        frozenset({2, 5}),
        frozenset({2, 6}),
    },
)
tree6_threebranch = (
    {1, 2, 3, 4, 5, 6},
    {
        frozenset({1, 2}),
        frozenset({2, 3}),
        frozenset({3, 4}),
        frozenset({3, 5}),
        frozenset({5, 6}),
    },
)

tree7_path = (
    {1, 2, 3, 4, 5, 6, 7},
    {
        frozenset({1, 2}),
        frozenset({2, 3}),
        frozenset({3, 4}),
        frozenset({4, 5}),
        frozenset({5, 6}),
        frozenset({6, 7}),
    },
)

graphs = {
    "tree3_path": tree3,
    "tree4_path": tree4_path,
    "tree4_star": tree4_star,
    "tree5_path": tree5_path,
    "tree5_star": tree5_star,
    "tree5_fork": tree5_fork,
    "tree6_star": tree6_star,
    "tree6_path": tree6_path,
    "tree6_fork": tree6_fork,
    "tree6_doublefork": tree6_doublefork,
    "tree6_lollipop": tree6_lollipop,
    "tree6_threebranch": tree6_threebranch,
    "tree7_path": tree7_path,
}
