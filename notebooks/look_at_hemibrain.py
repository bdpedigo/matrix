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
        "sandbox/results/look_at_hemibrain/" + name + f".{format}",
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


def symmetrze_nx(g):
    """Leiden requires a symmetric/undirected graph. This converts a directed graph to
    undirected just for this community detection step"""
    sym_g = nx.Graph()
    for source, target, weight in g.edges.data("weight"):
        if sym_g.has_edge(source, target):
            sym_g[source][target]["weight"] = (
                sym_g[source][target]["weight"] + weight * 0.5
            )
        else:
            sym_g.add_edge(source, target, weight=weight * 0.5)
    return sym_g


sym_g = symmetrze_nx(g)

#%%
adjacency = nx.to_scipy_sparse_matrix(sym_g, nodelist=list(g.nodes))

#%% [markdown]
# ## Create a class for recursive partitioning
#%%
class BaseNetworkTree(NodeMixin, BaseEstimator):
    def __init__(
        self,
        min_split=4,
        max_levels=4,
        verbose=False,
    ):
        self.min_split = min_split
        self.max_levels = max_levels
        self.verbose = verbose

    @property
    def node_data(self):
        if self.is_root:
            return self._node_data
        else:
            return self.root.node_data.loc[self._index]

    def _check_node_data(self, adjacency, node_data=None):
        if node_data is None and self.is_root:
            node_data = pd.DataFrame(index=range(adjacency.shape[0]))
            node_data["adjacency_index"] = range(adjacency.shape[0])
            self._node_data = node_data
            self._index = node_data.index

    def fit(self, adjacency, node_data=None):
        self._check_node_data(adjacency, node_data)

        if self.check_continue_splitting(adjacency):
            if self.verbose > 0:
                print(
                    f"[Depth={self.depth}, Number of nodes={adjacency.shape[0]}] Splitting subgraph..."
                )
            partition_labels = self._fit_partition(adjacency)
            self._split(adjacency, partition_labels)

        return self

    def check_continue_splitting(self, adjacency):
        return adjacency.shape[0] >= self.min_split and self.depth < self.max_levels

    def _split(self, adjacency, partition_labels):
        index = self._index
        node_data = self.root.node_data
        label_key = f"labels_{self.depth}"
        if label_key not in node_data.columns:
            node_data[label_key] = pd.Series(
                data=len(node_data) * [None], dtype="Int64"
            )

        unique_labels = np.unique(partition_labels)
        if self.verbose > 0:
            print(
                f"[Depth={self.depth}, Number of nodes={adjacency.shape[0]}] Split into {len(unique_labels)} groups"
            )
        if len(unique_labels) > 1:
            for i, label in enumerate(unique_labels):
                mask = partition_labels == label
                sub_adjacency = adjacency[np.ix_(mask, mask)]
                self.root.node_data.loc[index[mask], f"labels_{self.depth}"] = i
                # sub_node_data = self.node_data.loc[index[mask]]
                child = self.__class__(**self.get_params())
                child.parent = self
                child._index = index[mask]
                child.fit(sub_adjacency)

    @abstractmethod
    def _fit_partition(self, adjacency):
        pass

    def _hierarchical_mean(self, key):
        if self.is_leaf:
            index = self.node_data.index
            var = self.root.node_data.loc[index, key]
            return np.mean(var)
        else:
            children = self.children
            child_vars = [child._hierarchical_mean(key) for child in children]
            return np.mean(child_vars)


class LeidenTree(BaseNetworkTree):
    def __init__(
        self,
        trials=1,
        resolution=1.0,
        min_split=32,
        max_levels=4,
        verbose=False,
    ):
        super().__init__(
            min_split=min_split,
            max_levels=max_levels,
            verbose=verbose,
        )
        self.trials = trials
        self.resolution = resolution

    def _fit_partition(self, adjacency):
        """Fits a partition to the current subgraph using Leiden"""
        partition_map = leiden(adjacency, trials=self.trials)
        partition_labels = np.vectorize(partition_map.get)(
            np.arange(adjacency.shape[0])
        )
        return partition_labels


#%% [markdown]
# ## Fit to our data
#%%
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
ax, divider = adjplot(
    sorted_adjacency, plot_type="scattermap", sizes=(0.01, 0.01), ax=ax
)
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
