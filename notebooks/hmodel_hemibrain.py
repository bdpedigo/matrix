#%% [markdown]
# # Hierarchical partitioning and plotting on the hemibrain connectome
# > Here I demonstrate a simple hierarchical partitioner and how we can plot these results
#
# - toc: true
# - badges: false
# - categories: [pedigo, graspologic, connectome]
# - hide: false
# - search_exclude: false

#%%
import datetime
import time
from abc import abstractmethod
from pathlib import Path

import anytree
import colorcet as cc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from anytree import NodeMixin
from giskard.plot import graphplot, set_theme
from graspologic.partition import leiden
from graspologic.plot import adjplot
from sklearn.base import BaseEstimator

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
        "sandbox/results/hmodel_hemibrain/" + name + f".{format}",
        format=format,
        dpi=dpi,
        bbox_inches=bbox_inches,
        transparent=transparent,
        pad_inches=pad_inches,
        facecolor=facecolor,
    )


#%% [markdown]
# ## Load the raw data
# A link to the data is [here](https://storage.cloud.google.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz).
# More info about the project is [here](https://www.janelia.org/project-team/flyem/hemibrain).
#%%
data_path = Path("./sandbox/data/hemibrain/exported-traced-adjacencies-v1.2")

neuron_file = "traced-neurons.csv"
edgelist_file = "traced-total-connections.csv"

edgelist_df = pd.read_csv(data_path / edgelist_file)
edgelist_df = edgelist_df.rename(
    columns=dict(bodyId_pre="source", bodyId_post="target")
)
g = nx.from_pandas_edgelist(
    edgelist_df,
    source="source",
    target="target",
    edge_attr="weight",
    create_using=nx.DiGraph,
)

nodes = pd.read_csv(data_path / neuron_file)
#%% [markdown]
### Make the network undirected
#%%


def symmetrze_nx(g, weight="weight"):
    """Leiden requires a symmetric/undirected graph. This converts a directed graph to
    undirected just for this community detection step"""
    sym_g = nx.Graph()
    for source, target, weight_value in g.edges.data(weight):
        # TODO add support for all node and edge attributes as well
        # though, not sure it is worth worrying about edge attributes since they can
        # be anything.
        if sym_g.has_edge(source, target):
            sym_g[source][target][weight] = (
                sym_g[source][target][weight] + weight_value * 0.5
            )
        else:
            sym_g.add_edge(source, target, weight=weight_value * 0.5)
    return sym_g


sym_g = symmetrze_nx(g)

#%%
adjacency = nx.to_scipy_sparse_matrix(sym_g, nodelist=list(g.nodes)[:3000])

#%% [markdown]
# ## Fit to our data
#%%
from giskard.hierarchy import LeidenTree

lt = LeidenTree(verbose=True, max_levels=2)
lt.fit(adjacency)

#%% [markdown]
# ## Look at the sizes of leaf clusters
#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.histplot(x=[len(leaf.node_data) for leaf in lt.leaves], bins=20, ax=ax)
ax.set_xlabel("Size of leaf community")

#%% [markdown]
# ## Write some functions for plotting the dendrograms
# %%
def get_x_y(xs, ys, orientation):
    if orientation == "h":
        return xs, ys
    elif orientation == "v":
        return (ys, xs)


def plot_dendrogram(
    ax,
    root,
    index_key="sorted_adjacency_index",
    orientation="h",
    linewidth=0.7,
    cut=None,
    lowest_level=None,
):
    if lowest_level is None:
        lowest_level = root.height

    for node in (root.descendants) + (root,):
        y = node._hierarchical_mean(index_key)
        x = node.depth
        node.y = y
        node.x = x

    walker = anytree.Walker()
    walked = []

    for node in root.leaves:
        upwards, common, downwards = walker.walk(node, root)
        curr_node = node
        for up_node in (upwards) + (root,):
            edge = (curr_node, up_node)
            if edge not in walked:
                xs = [curr_node.x, up_node.x]
                ys = [curr_node.y, up_node.y]
                xs, ys = get_x_y(xs, ys, orientation)
                ax.plot(
                    xs,
                    ys,
                    linewidth=linewidth,
                    color="black",
                    alpha=1,
                )
                walked.append(edge)
            curr_node = up_node
        y_max = node.node_data[index_key].max()
        y_min = node.node_data[index_key].min()
        xs = [node.x, node.x, node.x + 1, node.x + 1]
        ys = [node.y - 3, node.y + 3, y_max, y_min]
        xs, ys = get_x_y(xs, ys, orientation)
        ax.fill(xs, ys, facecolor="black")

    if orientation == "h":
        ax.set(xlim=(-1, lowest_level + 1))
        if cut is not None:
            ax.axvline(cut - 1, linewidth=1, color="grey", linestyle=":")
    elif orientation == "v":
        ax.set(ylim=(lowest_level + 1, -1))
        if cut is not None:
            ax.axhline(cut - 1, linewidth=1, color="grey", linestyle=":")

    ax.axis("off")


#%% [markdown]
# ## Plot the network as an adjacency matrix
#%%
node_data = lt.node_data
node_data.sort_values(["labels_0", "labels_1"], inplace=True)
node_data["sorted_adjacency_index"] = range(len(node_data))
sorted_adjacency = adjacency[
    np.ix_(node_data["adjacency_index"], node_data["adjacency_index"])
]
fig, ax = plt.subplots(1, 1, figsize=(16, 16))
ax, divider = adjplot(sorted_adjacency, plot_type="scattermap", sizes=(0.1, 0.1), ax=ax)
left_ax = divider.append_axes("left", size="10%", pad=0, sharey=ax)
plot_dendrogram(left_ax, lt, orientation="h")

top_ax = divider.append_axes("top", size="10%", pad=0, sharex=ax)
plot_dendrogram(top_ax, lt, orientation="v")

stashfig("hleiden-adjplot")

#%% [markdown]
# ## Create a flat set of labels (leafs of the hierarchical clustering tree) for plotting
#%%
node_data = lt.node_data.copy()
node_data = node_data.set_index(["labels_0", "labels_1"])
flat_labels = node_data.index.to_flat_index()
node_data["labels_flat"] = flat_labels
palette = dict(zip(np.unique(flat_labels), cc.glasbey_light))

#%% [markdown]
# ## Plot the network as a 2D layout
#%%
plot_kws = dict(
    edge_linewidth=0.1,
    edge_alpha=0.1,
    subsample_edges=0.05,
    figsize=(12, 12),
    sizes=(3, 10),
    verbose=True,
)
graphplot(
    network=sorted_adjacency,
    meta=node_data.reset_index(),
    hue="labels_flat",
    node_palette=palette,
    **plot_kws,
)

#%%

from giskard.hierarchy import LeidenTree
adjacency = nx.to_scipy_sparse_matrix(sym_g, nodelist=list(g.nodes)[:5000], weight=None)
lt = LeidenTree(verbose=True, max_levels=2)
lt.fit(adjacency)
lt.estimate_parameters(adjacency)
probability_matrix = lt.construct_probability_matrix().values



#%%
node_data = lt.node_data
node_data.sort_values(["labels_0", "labels_1"], inplace=True)
node_data["sorted_adjacency_index"] = range(len(node_data))
sorted_adjacency = adjacency[
    np.ix_(node_data["adjacency_index"], node_data["adjacency_index"])
]
sorted_probabilitiy_matrix = probability_matrix[
    np.ix_(node_data["adjacency_index"], node_data["adjacency_index"])
]
fig, ax = plt.subplots(1, 1, figsize=(16, 16))

_, divider = adjplot(sorted_probabilitiy_matrix, plot_type="heatmap", ax=ax)
_, divider = adjplot(sorted_adjacency, plot_type="scattermap", sizes=(0.05, 0.05), ax=ax)


left_ax = divider.append_axes("left", size="10%", pad=0, sharey=ax)
plot_dendrogram(left_ax, lt, orientation="h")

top_ax = divider.append_axes("top", size="10%", pad=0, sharex=ax)
plot_dendrogram(top_ax, lt, orientation="v")

stashfig("hleiden-adjplot-modeled")

# #%%
# edgelist_df["source_group"] = edgelist_df["source"].map(partition_map)
# edgelist_df["target_group"] = edgelist_df["target"].map(partition_map)

# #%%
# block_strengths = pd.crosstab(
#     edgelist_df["source_group"],
#     edgelist_df["target_group"],
#     values=edgelist_df["weight"],
#     aggfunc=np.sum,
#     dropna=False,
# )

# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
# sns.heatmap(
#     block_strengths,
#     cmap="RdBu_r",
#     center=0,
#     square=True,
#     ax=ax,
#     cbar_kws=dict(shrink=0.7),
# )

#%%
elapsed = time.time() - t0
delta = datetime.timedelta(seconds=elapsed)
print("----")
print(f"Script took {delta}")
print(f"Completed at {datetime.datetime.now()}")
print("----")
