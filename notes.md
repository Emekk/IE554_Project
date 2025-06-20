# Remarks 
1. For bipartite graphs, there exists a dominator partition of order 2 if and only if the graph is a **complete** bipartite graph.
2. There is no dominator partition for **disconnected** graphs.
3. Adding edges to a graph would not change the feasibility of the current solution, it may add more dominators to the solution.

# Non-integral Solution Search
- all solutions are integral for:
  - tree3_path
  - tree4_path
      - k=2 