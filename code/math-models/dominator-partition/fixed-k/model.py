import gurobipy as gp
from gurobipy import GRB
import numpy as np
from draw_graph import draw_graph
import os


def create_and_solve_model(V, E, CN, MAXIMAL_INDEPENDENT_SETS, K, PI, SEARCH_FEASIBLE=False, VARIABLE_TYPE=GRB.BINARY, ALPHA=None, BETA=None):
    # model
    m = gp.Model('dominator-partition-fixed-k')

    # write to ssd instead of ram to prevent out of memory error
    # m.setParam('NodefileStart', 0.5)
    # m.setParam('Threads', 2)
    # m.setParam("NodefileDir", "C:\\Temp")
    m.setParam('OutputFlag', 0)

    # decision variables
    x = m.addVars(V, PI, vtype=VARIABLE_TYPE, lb=0, ub=1, name="x")  # x[v, i]
    d = m.addVars(V, PI, vtype=VARIABLE_TYPE, lb=0, ub=1, name="d")  # d[v, i]

    # objective: minimize number of blocks used
    if ALPHA is not None and BETA is not None:
        m.setObjective(
            gp.quicksum(ALPHA[v-1, i-1] * x[v, i] for v in V for i in PI) +
            gp.quicksum(BETA[v-1, i-1] * d[v, i] for v in V for i in PI),
            GRB.MINIMIZE
        )
    else:
        m.setObjective(0, GRB.MINIMIZE)

    # each vertex assigned to exactly one block
    for v in V:
        m.addConstr(gp.quicksum(x[v, i] for i in PI) == 1, name=f"Assign_{v}")

    # no empty blocks
    for i in PI:
        m.addConstr(gp.quicksum(x[v, i] for v in V) >= 1, name=f"NonEmptyBlock_{i}")

    # domination condition
    for v in V:
        for u in V:
            if ({v, u} not in E) and (v != u):
                for i in PI:
                    m.addConstr(x[u, i] + d[v, i] <= 1, name=f"Dominate_{v}_{u}_{i}")

    # each vertex dominates at least one block
    for v in V:
        m.addConstr(gp.quicksum(d[v, i] for i in PI) == 1, name=f"DominateBlock_{v}")

    # blocks are used in order
    for i in PI.difference({K}):
        m.addConstr(gp.quicksum(x[v, i] for v in V) >= gp.quicksum(x[v, i + 1] for v in V), name=f"Order_{i}")

    # VALID INEQUALITIES
    m.addConstr(gp.quicksum(x[v, 1] for v in V) >= np.ceil(len(V)/K), name="Valid_Min_Assignment1")
    for i in PI:
        m.addConstr(gp.quicksum(x[v, i] for v in V) <= np.floor((len(V)-K+i)/i), name=f"Valid_Max_Assignment_{i}")
    for v in V:
        for i in PI:
            m.addConstr(d[v, i] >= gp.quicksum(x[u, i] for u in CN[v]) - (len(CN[v])- 1), name=f"Valid_Dominate_{v}_{i}")
    for v in V:
        for i in PI:
            m.addConstr(d[v, i] <= gp.quicksum(x[u, i] for u in CN[v]), name=f"Valid_Dominate_Upper_{v}_{i}")
    for MIS in MAXIMAL_INDEPENDENT_SETS:
        for i in PI:
            m.addConstr(gp.quicksum(d[v, i] for v in MIS) <= 1, name=f"Valid_MIS_{i}")
    for i in PI:
        for v in V:
            for MIS in MAXIMAL_INDEPENDENT_SETS:
                if v in MIS:
                    m.addConstr(gp.quicksum(x[u, i] for u in CN[v]) <= len(CN[v]) - 1 + d[v, i] - gp.quicksum(d[y, i] for y in MIS if y != v), name=f"Valid_MIS_Cover_{v}_{i}")

    # run the model
    if SEARCH_FEASIBLE:
        m.setParam(GRB.Param.PoolSearchMode, 2)
        m.setParam(GRB.Param.PoolSolutions, 100)
    m.optimize()

    return {"model": m, "x": x, "d": d}

def display_results(m, x, d, V, E, PI, search_feasible, save_path, graph_path=None, partitioning=False):
    partitions = []
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(f"{save_path}", "w", encoding="utf-8") as f:
        if search_feasible:
            if m.SolCount > 0:
                print(f"Found {m.SolCount} solutions.\n", file=f)
                for solNum in range(m.SolCount):
                    m.setParam(GRB.Param.SolutionNumber, solNum)
                    print(f"Solution {solNum + 1}", file=f)
                    # print decision variables
                    for i in PI:
                        partition = []
                        for v in V:
                            if x[v, i].Xn > 0:
                                print(f"x[{v}, {i}] = {x[v, i].Xn}", file=f)
                                partition.append(v)
                        partitions.append(partition)
                    print("---", file=f)
                    for v in V:
                        for i in PI:
                            if d[v, i].Xn > 0:
                                print(f"d[{v}, {i}] = {d[v, i].Xn}", file=f)
                    print("", file=f)
            else:
                print("No feasible solution found.", file=f)
        elif m.status == GRB.OPTIMAL:
            for i in PI:
                partition = []
                for v in V:
                    if x[v, i].X > 0:
                        print(f"x[{v}, {i}] = {x[v, i].X}", file=f)
                        partition.append(v)
                partitions.append(partition)
            print("---", file=f)
            for v in V:
                for i in PI:
                    if d[v, i].X > 0:
                        print(f"d[{v}, {i}] = {d[v, i].X}", file=f)
        else:
            print("No feasible solution found.", file=f)
            
    if partitioning:
        draw_graph(V, E, partitions=partitions, seed=1, save_path=graph_path)

def evaluate_function_at_all_solutions(m, x, d, V, PI, coef_x, coef_d):
    min_value = np.inf
    max_value = -np.inf
    if m.SolCount > 0:
        for solNum in range(m.SolCount):
            m.setParam(GRB.Param.SolutionNumber, solNum)
            value = sum(coef_x[v-1, i-1] * x[v, i].Xn for v in V for i in PI) + sum(coef_d[v-1, i-1] * d[v, i].Xn for v in V for i in PI)
            if value < min_value:
                min_value = value
            if value > max_value:
                max_value = value
    return min_value, max_value


def format_term(coefficient, var_name, row_idx, col_idx):
    """Formats a single term of the inequality."""
    if coefficient == 0:
        return ""

    # Using 1-based indexing for display
    indices = f"{{{row_idx + 1},{col_idx + 1}}}"

    if coefficient == 1:
        coeff_str = ""
    elif coefficient == -1:
        coeff_str = "-"
    else:
        # Use a space for positive coefficients for alignment
        sign = "+ " if coefficient > 0 else "- "
        coeff_str = f"{abs(coefficient)} * "

    return f"{coeff_str}{var_name}_{indices}"

def display_elementwise_sum_inequality(coef_x, coef_d, operator="<=", rhs="0"):
    """
    Displays a single inequality formed by the element-wise product sum
    of coefficient matrices and variable matrices.
    """
    all_terms = []

    # Iterate through every element of the coef_x matrix
    for (v, i), coeff in np.ndenumerate(coef_x):
        term_str = format_term(coeff, 'x', v, i)
        if term_str:  # Only add non-zero terms
            all_terms.append(term_str)

    # Iterate through every element of the coef_d matrix
    for (v, i), coeff in np.ndenumerate(coef_d):
        term_str = format_term(coeff, 'd', v, i)
        if term_str:  # Only add non-zero terms
            all_terms.append(term_str)

    if not all_terms:
        return f"0 {operator} {rhs}"

    # Join all the collected terms and clean up the formatting
    expression = " + ".join(all_terms)
    expression = expression.replace('+ -', '- ') # Makes '... + -x' into '... - x'

    return f"{expression} {operator} {rhs}"
