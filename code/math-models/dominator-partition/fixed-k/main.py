from sample_graphs import *
import numpy as np
from model import create_and_solve_model, display_results
import time


# parameters
SEED = 0
N = 10  # Number of nodes
P = 0.2  # Probability of edge creation
#K = max(int(N - np.ceil((P * (N**2 - N) / 2))**(0.6)), int(N**(0.5)))
K = 4
GRAPH_NAME = f"random_graph{N}_{P}"
#GRAPH_NAME = "tree7_path"
SEARCH_FESAIBLE = False
V, E = build_random_graph(N, P, seed=SEED)
#V, E = graphs[GRAPH_NAME]
CN = closed_neighborhoods(V, E)
MAXIMAL_INDEPENDENT_SETS = list(get_maximal_independent_sets(V, E))
print(f"Maximal Independent Sets: {MAXIMAL_INDEPENDENT_SETS}")
PI = {i for i in range(1, K+1)}  # Number of blocks (fixed)
print(f"N={N}, P={P}, K={K}")

print(f">>> Graph: {GRAPH_NAME}, K: {K} - Model starting...")
t = time.time()
# RUN THE MODEL ONE TIME
m, x, d = create_and_solve_model(
    V=V,
    E=E,
    CN=CN,
    MAXIMAL_INDEPENDENT_SETS=MAXIMAL_INDEPENDENT_SETS,
    K=K,
    PI=PI,
    SEARCH_FESAIBLE=SEARCH_FESAIBLE,
    ALPHA = None,  
    BETA = None
).values()
print(f"Model took: {time.time() - t:,.2f}")

display_results(m, x, d, V, E, PI, SEARCH_FESAIBLE, save_path=f"solutions/{GRAPH_NAME}/k={K}/seed={SEED}.txt", partitioning=True)

# is_integral = True
# is_feasible = True
# while is_integral: 
#     # generate random weights for x[v, i] and d[v, i]
#     SEED += 1
#     np.random.seed(SEED)
#     # ALPHA = np.random.rand(len(V), len(PI)) * 2 - 1  # Random weights for x[v, i]
#     # BETA = np.random.rand(len(V), len(PI)) * 2 - 1  # Random weights for d[v, i]

#     # create and solve model
#     m, x, d = create_and_solve_model(
#         V=V,
#         E=E,
#         CN=CN,
#         MAXIMAL_INDEPENDENT_SETS=MAXIMAL_INDEPENDENT_SETS,
#         K=K,
#         PI=PI,
#         ALPHA=ALPHA,
#         BETA=BETA,
#         SEARCH_FESAIBLE=SEARCH_FESAIBLE
#     ).values()
    
#     if m.status == GRB.INFEASIBLE:
#         print(f">>> Model is infeasible with seed {SEED}.")
#         break

#     # check if the solution is integral
#     is_integral = all(x[v, i].X.is_integer() and d[v, i].X.is_integer() for v in V for i in PI)

#     if SEED % 1000 == 0:
#         print(SEED)

# if is_feasible and not is_integral:
#     print(f">>> Found non-integral solution with seed {SEED}.")
#     display_results(m, x, d, V, E, PI, SEARCH_FESAIBLE, save_path=f"solutions/{GRAPH_NAME}/k={K}/seed={SEED}.txt")
#     draw_graph(V, E, save_path=f"solutions/{GRAPH_NAME}/graph.png")
