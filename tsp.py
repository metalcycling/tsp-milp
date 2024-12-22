# %% Modules

import itertools
import networkx as nx
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.optimize import milp, LinearConstraint, Bounds

# %% Problem parameters

num_nodes = 8
nodes = [node for node in range(1, num_nodes + 1)]

edges = list(itertools.permutations(nodes, 2))
num_edges = len(edges)

# %% Mapping

var_to_idx = {}
idx_to_var = {}
idx = 0

for edge in edges:
    var_to_idx[edge] = idx
    idx_to_var[idx] = edge
    idx += 1

for node in nodes[1:]:
    var_to_idx[node] = idx
    idx_to_var[idx] = node
    idx += 1

# %% Cost

distances = np.ones((num_nodes, num_nodes), dtype = float)
np.fill_diagonal(distances, 0.0)

distances[0, 1] = 10.0
distances[1, 2] = 10.0
distances[3, 4] = 10.0
distances[2, 3] = 10.0
distances[0, 4] = 10.0

# %% Edges without first nodes

sub_edges = list(itertools.permutations(nodes[1:], 2))
num_sub_edges = len(sub_edges)

# %% Objective function

objective_coefficiens = []

for edge in edges:
    objective_coefficiens.append(distances[edge[0] - 1, edge[1] - 1])

objective_coefficiens = np.hstack((
    np.array(objective_coefficiens),
    np.zeros(num_nodes - 1),
))

# %% Constraints

constraints = {
      "A": np.zeros((2 * num_nodes + num_sub_edges + num_nodes - 1, num_edges + num_nodes - 1)),
    "b_l": np.hstack((np.ones(2 * num_nodes), np.full(num_sub_edges,                - np.inf), np.full(num_nodes - 1,       2.0))),
    "b_u": np.hstack((np.ones(2 * num_nodes), np.full(num_sub_edges, (num_nodes - 1.0) - 1.0), np.full(num_nodes - 1, num_nodes))),
}

cdx = 0

for node_i in nodes:
    for node_j in [node for node in nodes if node != node_i]:
        constraints["A"][cdx + 0, var_to_idx[(node_i, node_j)]] = 1.0
        constraints["A"][cdx + 1, var_to_idx[(node_j, node_i)]] = 1.0

    cdx += 2

for edge in sub_edges:
    constraints["A"][cdx, var_to_idx[edge]] = num_nodes - 1.0
    constraints["A"][cdx, var_to_idx[edge[0]]] = 1.0
    constraints["A"][cdx, var_to_idx[edge[1]]] = - 1.0
    cdx += 1

for node in nodes[1:]:
    constraints["A"][cdx, var_to_idx[node]] = 1.0
    cdx += 1

constraints = LinearConstraint(constraints["A"], constraints["b_l"], constraints["b_u"])

# %% Set bounds for the variables

bounds = Bounds(
    lb = np.hstack((np.zeros(num_edges), np.zeros(num_nodes - 1))),
    ub = np.hstack((np.ones(num_edges), np.full(num_nodes - 1, num_nodes))),
)

integrality = np.ones_like(objective_coefficiens)

# %% Solve the MILP problem

solution = milp(c = objective_coefficiens, constraints = constraints, integrality = integrality, bounds = bounds)
path = solution.x

# %%


# %% Visualize solution

G = nx.DiGraph()
G.add_nodes_from(nodes)

for edge in edges:
    node_i = edge[0]
    node_j = edge[1]

    if distances[node_i - 1, node_j - 1] == 1.0:
        G.add_edge(edge[0], edge[1])

P = nx.DiGraph()

for idx in np.where(path[:num_edges])[0]:
    node_i = idx_to_var[idx][0]
    node_j = idx_to_var[idx][1]

    P.add_edge(node_i, node_j)

pos = nx.circular_layout(G)

graph_options = {
    "font_size": 36,
    "node_size": 3000,
    "edgecolors": "black",
    "edge_color": "lightgray",
    "linewidths": 5,
    "width": 5,
}

path_options = {
    "font_size": 36,
    "node_size": 3000,
    "node_color": "cornflowerblue",
    "edgecolors": "black",
    "edge_color": "red",
    "linewidths": 5,
    "width": 5,
}

plt.figure(figsize = (12, 8))
nx.draw_networkx(G, pos = pos, **graph_options)
nx.draw_networkx(P, pos = pos, **path_options)
plt.axis("equal")
plt.show()

# %% End of script
