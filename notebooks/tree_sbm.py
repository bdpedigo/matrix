#%%
import time

import matplotlib.pyplot as plt
import numpy as np
from giskard.hierarchy import LeidenTree
from giskard.plot import plot_dendrogram, set_theme
from graspologic.plot import adjplot
from graspologic.simulations import sbm

t0 = time.time()
set_theme()


def stashfig(
    name,
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.1,
    transparent=False,
    facecolor="white",
    **kws,
):
    plt.savefig(
        "sandbox/results/tree_sbm/" + name + f".{format}",
        format=format,
        dpi=dpi,
        bbox_inches=bbox_inches,
        transparent=transparent,
        pad_inches=pad_inches,
        facecolor=facecolor,
    )


#%%
B = np.array(
    [
        [0.6, 0.3, 0.3, 0.1, 0.1, 0.1],
        [0.3, 0.6, 0.3, 0.1, 0.1, 0.1],
        [0.3, 0.3, 0.6, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.6, 0.3, 0.3],
        [0.1, 0.1, 0.1, 0.3, 0.6, 0.3],
        [0.1, 0.1, 0.1, 0.3, 0.3, 0.6],
    ]
)

n_per_comm = 100
ns = 6 * [n_per_comm]

n_trials = 20
probability_matrix = np.zeros((600, 600))
for _ in range(n_trials):
    adjacency, labels = sbm(ns, B, return_labels=True)
    lt = LeidenTree(trials=5, verbose=False, max_levels=2)
    lt.fit(adjacency)
    lt.estimate_parameters(adjacency)
    probability_matrix += lt.full_probability_matrix.values / n_trials

np.unique(probability_matrix)

#%%

node_data = lt.node_data
node_data.sort_values(["labels_0", "labels_1"], inplace=True)
node_data["sorted_adjacency_index"] = range(len(node_data))
sorted_adjacency = adjacency[
    np.ix_(node_data["adjacency_index"], node_data["adjacency_index"])
]
fig, ax = plt.subplots(1, 1, figsize=(16, 16))
ax, divider = adjplot(sorted_adjacency, plot_type="heatmap", ax=ax)
left_ax = divider.append_axes("left", size="10%", pad=0, sharey=ax)
plot_dendrogram(left_ax, lt, orientation="h")

top_ax = divider.append_axes("top", size="10%", pad=0, sharex=ax)
plot_dendrogram(top_ax, lt, orientation="v")
