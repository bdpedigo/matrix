#%%
import time
from graspologic.simulations.simulations import er_np, sample_edges

import matplotlib.pyplot as plt
import numpy as np
from giskard.hierarchy import LeidenTree
from giskard.plot import plot_dendrogram, set_theme
from graspologic.plot import adjplot
from graspologic.simulations import sbm
import pandas as pd

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
        "sandbox/results/tree_sbm_explain/" + name + f".{format}",
        format=format,
        dpi=dpi,
        bbox_inches=bbox_inches,
        transparent=transparent,
        pad_inches=pad_inches,
        facecolor=facecolor,
    )


#%%

import anytree

zero = anytree.Node(0)
one = anytree.Node(1, parent=zero)
two = anytree.Node(2, parent=zero)
three = anytree.Node(3, parent=one)
four = anytree.Node(4, parent=one)
five = anytree.Node(5, parent=two)
six = anytree.Node(6, parent=two)
seven = anytree.Node(7, parent=two)
eight = anytree.Node(8, parent=three)
nine = anytree.Node(9, parent=three)
root = zero

for pre, _, node in anytree.RenderTree(root):
    print("%s%s" % (pre, node.name))


#%%
unique_paths = np.array(
    [
        [0, 2, 7, np.nan],
        [0, 2, 6, np.nan],
        [0, 2, 5, np.nan],
        [0, 1, 4, np.nan],
        [0, 1, 3, 9],
        [0, 1, 3, 8],
    ]
)

n_per_leaf = [10, 10, 20, 10, 5, 5]

paths = []
for path, n_in_leaf in zip(unique_paths, n_per_leaf):
    for i in range(n_in_leaf):
        paths.append(list(path))

meta = pd.DataFrame(paths, columns=[f"lvl{i}" for i in range(4)])
meta["adjacency_index"] = range(len(meta))
#%%
from giskard.hierarchy import MetaTree
import seaborn as sns

mt = MetaTree(max_levels=np.inf, min_split=0)
mt.build(meta, "lvl", offset=1)
mt.name = 0


for pre, _, node in anytree.RenderTree(mt):
    node.name = int(node.name)
    print("%s%s" % (pre, len(node._index)))

probs = {
    0: 0.05,
    1: 0.1,
    2: 0.15,
    3: 0.2,
    4: 0.3,
    5: 0.25,
    6: 0.3,
    7: 0.4,
    8: 0.6,
    9: 0.6,
}

for node in anytree.PreOrderIter(mt):
    node.probability_estimate_ = probs[node.name]

mat = mt.full_probability_matrix.values

from graspologic.simulations import sample_edges

adj = sample_edges(mat, directed=False, loops=False)

#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
from graspologic.simulations import er_np

# adj = er_np(sum(n_per_leaf), 0.1)

_, divider = adjplot(mat, plot_type="heatmap", ax=ax)

top_ax = divider.append_axes("top", size="40%", pad=0, sharex=ax)
plot_dendrogram(
    top_ax,
    mt,
    index_key="adjacency_index",
    orientation="v",
    linewidth=2,
    markersize=25,
    linecolor="grey",
    markercolor="white",
)

left_ax = divider.append_axes("left", size="40%", pad=0, sharey=ax)
plot_dendrogram(
    left_ax,
    mt,
    index_key="adjacency_index",
    orientation="h",
    linewidth=2,
    markersize=25,
    linecolor="grey",
    markercolor="white",
)

#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
from graspologic.simulations import er_np

# adj = er_np(sum(n_per_leaf), 0.1)
_, divider = adjplot(adj, plot_type="heatmap", ax=ax)

top_ax = divider.append_axes("top", size="40%", pad=0, sharex=ax)
plot_dendrogram(
    top_ax,
    mt,
    index_key="adjacency_index",
    orientation="v",
    linewidth=2,
    markersize=25,
    linecolor="grey",
    markercolor="white",
)

left_ax = divider.append_axes("left", size="40%", pad=0, sharey=ax)
plot_dendrogram(
    left_ax,
    mt,
    index_key="adjacency_index",
    orientation="h",
    linewidth=2,
    markersize=25,
    linecolor="grey",
    markercolor="white",
)

#%%


def nearest_common_ancestor(source, target):
    walker = anytree.Walker()
    _, nearest_common_ancestor, _ = walker.walk(source, target)
    return nearest_common_ancestor


n_leaves = len(root.leaves)
leaves = [n.name for n in root.leaves]
sbm_probs = np.zeros((n_leaves, n_leaves))
sbm_probs = pd.DataFrame(data=sbm_probs, index=leaves, columns=leaves)
alpha = 0.7
for source_node in anytree.PreOrderIter(root):
    for target_node in anytree.PreOrderIter(root):
        if source_node.is_leaf and target_node.is_leaf:
            nca = nearest_common_ancestor(source_node, target_node).name
            base_prob = probs[nca]
            new_prob = np.random.uniform(
                base_prob - alpha * base_prob, base_prob + alpha * base_prob
            )
            i = source_node.name
            j = target_node.name
            sbm_probs.loc[i, j] = new_prob

from graspologic.utils import symmetrize

sbm_probs = sbm_probs.values
sbm_probs = symmetrize(sbm_probs)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
adjplot(sbm_probs, ax=ax)

# %%
flat_labels = []
node_data = mt.node_data
for node, row in node_data.iterrows():
    path = row.values[:4]
    path = path[~np.isnan(path)]
    label = path[-1]
    flat_labels.append(label)

flat_labels = np.array(flat_labels)

#%%
A, flat_labels = sbm(n_per_leaf, sbm_probs, directed=False, return_labels=True)

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
adjplot(A, ax=ax)