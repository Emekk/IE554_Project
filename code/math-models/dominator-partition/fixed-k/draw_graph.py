import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from sample_graphs import build_random_graph, graphs


script_dir = os.path.dirname(os.path.abspath(__file__))

def draw_graph(V, E, partitions=None, save_path=f"{script_dir}/graph.png", seed=42, separation=10):
    """
    Draws a graph from an adjacency matrix and optionally colors the nodes based on partitions.
    
    Parameters:
    - adjacency_matrix: 2D numpy array representing the adjacency matrix of the graph.
    - partitions: List of lists, where each sublist contains the indices of nodes in that partition.
    """
    G = nx.Graph()
    G.add_nodes_from(V)
    G.add_edges_from(E)

    pos = nx.spring_layout(G, seed=seed, k=separation)
    if partitions:
        # Assign a label to each node based on its partition
        node_labels = np.zeros(len(G.nodes()), dtype=int)
        for label, part in enumerate(partitions):
            for node in part:
                node_labels[node-1] = label

        cmap = cm.get_cmap('tab20', len(partitions))
        node_colors = [cmap(label) for label in node_labels]

        nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray',
                node_size=1000, font_size=16)
    else:
        nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=1000, font_size=16)

    plt.title("Graph Visualization")
    plt.show()
    plt.savefig(save_path, dpi=300)


if __name__ == "__main__":
    # Example usage
    V, E = build_random_graph(20, 0.4)  # Random graph with 10 vertices and edge probability 0.3
    V, E = graphs["tree5_path"]
    draw_graph(V, E, save_path=f"{script_dir}/example_graph.png")
