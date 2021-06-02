#%%
import time
import datetime
from pathlib import Path
from numpy.core.fromnumeric import partition
import pandas as pd
import networkx as nx
from giskard.plot import graphplot
from graspologic.partition import leiden
from graspologic.layouts.colors import _get_colors
import numpy as np
from graspologic.partition import hierarchical_leiden, HierarchicalCluster
# from src.io import savefig
from sklearn.model_selection import ParameterGrid

t0 = time.time()

main_random_state = np.random.default_rng(8888)


# def stashfig(name):
#     savefig(name, foldername="hemibrain-layout")


# TODO this is a bit buried in graspologic, should expose it more explicitly
colors = _get_colors(True, None)["nominal"]

data_path = Path("../data/hemibrain/exported-traced-adjacencies-v1.2")

neuron_file = "traced-neurons.csv"
edgelist_file = "traced-total-connections.csv"

edgelist_df = pd.read_csv(data_path / edgelist_file)
edgelist_df = edgelist_df.rename(columns=dict(bodyId_pre='source', bodyId_post='target'))
g = nx.from_pandas_edgelist(
    edgelist_df,
    source="source",
    target="target",
    edge_attr="weight",
    create_using=nx.DiGraph,
)

nodes = pd.read_csv(data_path / neuron_file)
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
partition_map = leiden(sym_g, trials=25)

#%%
edgelist_df['source_group'] = edgelist_df['source'].map(partition_map)
edgelist_df['target_group'] = edgelist_df['target'].map(partition_map)

#%%
block_strengths = pd.crosstab(edgelist_df['source_group'], edgelist_df['target_group'], values=edgelist_df['weight'], aggfunc=np.sum, dropna=False)
import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1,figsize=(8,8))
sns.heatmap(block_strengths, cmap='RdBu_r', center=0, square=True, ax=ax, cbar_kws=dict(shrink=0.7))

#%%
from anytree import NodeMixin
from abc import abstractmethod
from graspologic.utils import symmetrize

class BaseNetworkTree(NodeMixin):
    def __init__(self, resolution=1.0, min_split=4, max_levels=4, parent=None, children=[], verbose=False):
        self.min_split = min_split
        self.max_levels = max_levels
        self.children = children
        self.parent = parent
        self.verbose = verbose
        self.resolution = resolution
    
    @abstractmethod
    def fit(self, network):
        pass

    @abstractmethod
    def split(self, network):
        pass


class LeidenTree(BaseNetworkTree):
    def __init__(self, trials=1, min_split=None, max_levels=4, parent=None, children=[], verbose=False):
        super().__init__(min_split=min_split,max_levels=max_levels, parent=parent, children=children, verbose=verbose)
        self.trials = trials

    def fit(self, adjacency, node_data=None):
        if node_data is None: 
            node_data = pd.DataFrame(index=range(adjacency.shape[0]))
            node_data['adjacency_index'] = range(adjacency.shape[0])
        self.node_data = node_data

        min_split = self.min_split
        if min_split is None: 
            min_split = int(np.sqrt(adjacency.shape[0]))

        if adjacency.shape[0] >= min_split and self.depth < self.max_levels:
            if self.verbose > 0: 
                print(f"Splitting at depth {self.depth}")
            partition_labels = self.split(adjacency)
            self.dispatch(adjacency, partition_labels)

    def split(self, adjacency):
        partition_map = leiden(adjacency, trials=self.trials)
        partition_labels = np.vectorize(partition_map.get)(np.arange(adjacency.shape[0]))
        return partition_labels

    def dispatch(self, adjacency, partition_labels):
        unique_labels = np.unique(partition_labels)
        if self.verbose > 0: 
            print(f"Split into {len(unique_labels)} groups")
        for label in unique_labels:
            mask = partition_labels == label
            sub_adjacency = adjacency[np.ix_(mask, mask)]
            sub_node_data = self.node_data.iloc[mask]
            if self.verbose > 0: 
                print(f"Group has {sub_adjacency.shape[0]} nodes")
            child = LeidenTree(trials=self.trials, min_split=self.min_split, max_levels=self.max_levels, parent=self, verbose=self.verbose)
            child.fit(sub_adjacency, node_data=sub_node_data)

#%%
adjacency = nx.to_scipy_sparse_matrix(sym_g)
#%%
lt = LeidenTree(verbose=True, max_levels=2)
lt.fit(adjacency)

#%%

partition_map = leiden(adjacency)

#%%
partition_labels = np.vectorize(partition_map.get)(np.arange(adjacency.shape[0]))
unique_labels, partition_index_labels = np.unique(partition_labels, return_inverse=True)
for label in unique_labels:
    mask = partition_labels == label
    sub_adjacency = adjacency[np.ix_(mask, mask)]

#%%

[len(leaf.node_data) for leaf in lt.leaves]