from sample_graphs import *
import numpy as np
from model import create_and_solve_model, display_results, evaluate_function_at_all_solutions, display_elementwise_sum_inequality
from draw_graph import draw_graph
from gurobipy import GRB
import time


# parameters
SEED = 0
N = 4  # Number of nodes
SHAPE = "path"  # Shape of the tree: "path", "star", "fork"...
#P = 0.2  # Probability of edge creation
#K = max(int(N - np.ceil((P * (N**2 - N) / 2))**(0.6)), int(N**(0.5)))
K = 3
#GRAPH_NAME = f"random_graph{N}_{P}"
GRAPH_NAME = f"tree{N}_{SHAPE}"
#V, E = build_random_graph(N, P, seed=SEED)
V, E = graphs[GRAPH_NAME]
CN = closed_neighborhoods(V, E)
MAXIMAL_INDEPENDENT_SETS = list(get_maximal_independent_sets(V, E))
print(f"Maximal Independent Sets: {MAXIMAL_INDEPENDENT_SETS}")
PI = {i for i in range(1, K+1)}  # Number of blocks (fixed)
#print(f"N={N}, P={P}, K={K}")

print(f">>> Graph: {GRAPH_NAME}, K: {K} - Model starting...")
t = time.time()
#RUN THE MODEL ONE TIME
m_integer, x_integer, d_integer = create_and_solve_model(
    V=V,
    E=E,
    CN=CN,
    MAXIMAL_INDEPENDENT_SETS=MAXIMAL_INDEPENDENT_SETS,
    K=K,
    PI=PI,
    SEARCH_FEASIBLE=True,
    ALPHA = None,
    BETA = None
).values()
print(f"Model took: {time.time() - t:,.2f}")

display_results(m_integer, x_integer, d_integer, V, E, PI, True, save_path=f"solutions_integer/{GRAPH_NAME}/k={K}/seed={SEED}.txt", partitioning=False)

# ---

is_integral = True
is_feasible = True
while is_integral: 
    # generate random weights for x[v, i] and d[v, i]
    SEED += 1
    np.random.seed(SEED)
    ALPHA = np.random.rand(len(V), len(PI)) * 2 - 1  # Random weights for x[v, i]
    BETA = np.random.rand(len(V), len(PI)) * 2 - 1  # Random weights for d[v, i]

    # create and solve model
    m, x, d = create_and_solve_model(
        V=V,
        E=E,
        CN=CN,
        MAXIMAL_INDEPENDENT_SETS=MAXIMAL_INDEPENDENT_SETS,
        K=K,
        PI=PI,
        VARIABLE_TYPE=GRB.CONTINUOUS,
        ALPHA=ALPHA,
        BETA=BETA,
        SEARCH_FEASIBLE=False
    ).values()

    if m.status == GRB.INFEASIBLE:
        print(f">>> Model is infeasible with seed {SEED}.")
        break

    # check if the solution is integral
    is_integral = all(x[v, i].X.is_integer() and d[v, i].X.is_integer() for v in V for i in PI)

    if SEED % 1000 == 0:
        print(f"SEED: {SEED}")

if is_feasible and not is_integral:
    print(f">>> Found fractional solution with seed {SEED}.\n")
    display_results(m, x, d, V, E, PI, False, save_path=f"solutions/{GRAPH_NAME}/k={K}/seed={SEED}.txt")
    # valid inequality search
    seed = 0
    searching = True
    while searching:
        np.random.seed(seed)
        coef_x = np.random.choice(a=[-1, 0, 1], size=(len(V), len(PI)))
        coef_d = np.random.choice(a=[-1, 0, 1], size=(len(V), len(PI)))
        min_value, max_value = evaluate_function_at_all_solutions(m_integer, x_integer, d_integer, V, PI, coef_x=coef_x, coef_d=coef_d)
        value_fractional_solution = sum(coef_x[v-1, i-1] * x[v, i].X for v in V for i in PI) + sum(coef_d[v-1, i-1] * d[v, i].X for v in V for i in PI)
        if value_fractional_solution < min_value or value_fractional_solution > max_value:
            searching = False
            print(f"Found valid inequality with seed {seed}")
            print(f"Min value at integer solutions: {min_value}")
            print(f"Max value at integer solutions: {max_value}")
            print(f"Value at fractional solution: {value_fractional_solution}")
            print(f"coeff_x:\n{coef_x}")
            print(f"coeff_d:\n{coef_d}")
            print("Valid Inequality:")
            operator = ">=" if value_fractional_solution < min_value else "<="
            rhs = min_value if value_fractional_solution < min_value else max_value
            display_elementwise_sum_inequality(coef_x, coef_d, operator, rhs)
        seed += 1
