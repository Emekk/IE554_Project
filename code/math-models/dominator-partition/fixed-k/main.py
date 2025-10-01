from sample_graphs import *
import numpy as np
from model import create_and_solve_model, display_results, evaluate_function_at_all_solutions, display_elementwise_sum_inequality
from gurobipy import GRB
import itertools
import math
import time
import os


# parameters
fractional_seed = 0
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

display_results(m_integer, x_integer, d_integer, V, E, PI, True, save_path=f"solutions_integer/{GRAPH_NAME}/k={K}/seed={fractional_seed}.txt", partitioning=False)

# Search for fractional solutions and corresponding valid inequalities recursively
found_fractional_solutions = 0
fractional_solution_start_time = time.time()
while (found_fractional_solutions < 10) and (time.time() - fractional_solution_start_time < 150):
    is_integral = True
    is_feasible = True
    while is_integral: 
        # generate random weights for x[v, i] and d[v, i]
        fractional_seed += 1
        np.random.seed(fractional_seed)
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
            print(f">>> Model is infeasible with seed {fractional_seed}.")
            break

        # check if the solution is integral
        is_integral = all(x[v, i].X.is_integer() and d[v, i].X.is_integer() for v in V for i in PI)

    if is_feasible and not is_integral:
        found_fractional_solutions += 1
        print(f">>> Found fractional solution #{found_fractional_solutions} with seed {fractional_seed}.\n")
        fractional_solution_path = f"solutions_fractional/{GRAPH_NAME}/k={K}/seed={fractional_seed}.txt"
        display_results(m, x, d, V, E, PI, False, save_path=fractional_solution_path)
        # valid inequality search
        integer_seed = 0
        found_valid_inequalities = 0
        valid_inequality_start_time = time.time()
        # temp change start
        n = 2 * len(V) * len(PI)
        num_replacements = 12
        coef_xs = []
        coef_ds = []
        for k in range(1, num_replacements + 1):
            for combo in itertools.combinations(range(n), k):
                row = np.random.choice(a=[-1, 1], size=n)
                row[list(combo)] = 0
                coef_x = row[:n//2].reshape((len(V), len(PI)))
                coef_d = row[n//2:].reshape((len(V), len(PI)))
                coef_xs.append(coef_x)
                coef_ds.append(coef_d)
        print(f"Total combinations to check: {len(coef_xs)}")
        # temp change end
        while (found_valid_inequalities < 50) and (time.time() - valid_inequality_start_time < 300):
            integer_seed += 1
            np.random.seed(integer_seed)
            # coef_x = np.random.choice(a=[-1, 0, 1], size=(len(V), len(PI)))
            # coef_d = np.random.choice(a=[-1, 0, 1], size=(len(V), len(PI)))
            # temp change start
            coef_x = coef_xs[integer_seed]
            coef_d = coef_ds[integer_seed]
            # temp change end
            min_value, max_value = evaluate_function_at_all_solutions(m_integer, x_integer, d_integer, V, PI, coef_x=coef_x, coef_d=coef_d)
            value_fractional_solution = sum(coef_x[v-1, i-1] * x[v, i].X for v in V for i in PI) + sum(coef_d[v-1, i-1] * d[v, i].X for v in V for i in PI)
            if (not math.isclose(value_fractional_solution, min_value, abs_tol=1e-3) and value_fractional_solution < min_value) or\
                (not math.isclose(value_fractional_solution, max_value, abs_tol=1e-3) and value_fractional_solution > max_value):
                found_valid_inequalities += 1
                print(f"Found valid inequality #{found_valid_inequalities} with seed {integer_seed}")
                print(f"Min: {min_value}; Max: {max_value}; Fractional: {value_fractional_solution}")

                # save the valid inequality to a file
                operator = ">=" if value_fractional_solution < min_value else "<="
                rhs = min_value if value_fractional_solution < min_value else max_value
                valid_path = f"solutions_valid_inequalities/{GRAPH_NAME}/k={K}/fractional_seed={fractional_seed}.txt"
                valid_inequality_str = display_elementwise_sum_inequality(coef_x, coef_d, operator, rhs)
                if os.path.exists(valid_path):
                    with open(valid_path, "a", encoding="utf-8") as f:
                        print(f"\nValid inequality of integer_seed={integer_seed}:\n\t{valid_inequality_str}", file=f)
                else:
                    os.makedirs(os.path.dirname(valid_path), exist_ok=True)
                    with open(fractional_solution_path, "r", encoding="utf-8") as f_in, open(valid_path, "w", encoding="utf-8") as f_out:
                        f_out.write(f_in.read())
                        print(f"\nValid inequality of integer_seed={integer_seed}:\n\t{valid_inequality_str}", file=f_out)
        fractional_solution_start_time = time.time()
