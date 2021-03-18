#%%
import json
from collections import defaultdict

import colorcet as cc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text

from giskard.plot import graphplot, palplot
from graspologic.utils import largest_connected_component

years = np.arange(2015, 2022)


savefig_kws = dict(
    dpi=300, pad_inches=0.3, transparent=False, bbox_inches="tight", facecolor="white"
)


def stashfig(name, format="png", **kwargs):
    plt.savefig(f"sandbox/results/kavli/{name}.{format}", **savefig_kws)


class MapDict(dict):
    __missing__ = lambda self, key: key


def flatten_muligraph(multigraph, filter_edge_type=None):
    # REF: https://stackoverflow.com/questions/15590812/networkx-convert-multigraph-into-simple-graph-with-weighted-edges
    if isinstance(filter_edge_type, str):
        filter_edge_type = [filter_edge_type]
    g = nx.Graph()
    # for node in multigraph.nodes():
    #     g.add_node(node)
    for i, j, data in multigraph.edges(data=True):
        edge_type = data["type"]
        if filter_edge_type is None or edge_type in filter_edge_type:
            w = data["weight"] if "weight" in data else 1.0
            if g.has_edge(i, j):
                g[i][j]["weight"] += w
            else:
                g.add_edge(i, j, weight=w)
    return g


# load the data
data_loc = "sandbox/data/kavli/Kavli Interconnections 2021.xlsx"
data = pd.read_excel(data_loc)
data["source"] = list(data["source_first"] + " " + data["source_last"])
data["target"] = list(data["target_first"] + " " + data["target_last"])

# ignore anything having to do with "joined" for the purposes of the edgelist
edges = data[data["type"] != "joined"].copy()
# these are edges that are not edges?
bad_edges = edges[edges["source"].isna() | edges["target"].isna()]
edges = edges[~edges.index.isin(bad_edges.index)]


# create the actual network, using only the internal connections
mg = nx.from_pandas_edgelist(
    edges, edge_attr=["type", "ternal", "start", "end"], create_using=nx.MultiGraph
)
# some postprocessing
g = flatten_muligraph(mg)
g = largest_connected_component(g)


# some node metadata
targets = edges["target"].unique()
sources = edges["source"].unique()
node_ids = np.unique(np.array(list(targets) + list(sources)))
nodes = pd.DataFrame(index=node_ids)
nodes = nodes[nodes.index.isin(g.nodes)]
departments = edges.groupby("target")["target_dept"].agg(lambda x: x.mode()[0])
nodes["dept"] = nodes.index.map(departments).fillna("unk")

edge_types = edges["type"].unique()
joined = data[data["type"] == "joined"].set_index("source")
nodes["core"] = False
nodes.loc[nodes.index.isin(joined.index), "core"] = True
# nodes.loc[~nodes.index.isin(joined.index), "core"] = False

# mapping departments
dept_map = {
    "BME/JHM": "BME",
    "Mind Brain Institute": "MBI",
    "Mind Brain Institute, Neuroscience": "Neuroscience",
    "BME/Ophthalmology": "BME",
    "Neuro": "Neuroscience",
    "Neuro + BME": "Neuroscience",
    "Biostat": "Biostats",
    "ME": "Mechanical Engineering",
}

dept_map = MapDict(dept_map)

nodes["dept"] = nodes["dept"].map(dept_map)
nodes.loc["Peter Searson", "dept"] = "Materials Science and Engineering"
nodes["dept"].unique()


#%% [markdown]
# ## Create a color palette
#%%
colors = sns.color_palette("husl", nodes["dept"].nunique())
palette = dict(zip(sorted(nodes["dept"].unique()), colors))
nodes["color"] = nodes["dept"].map(palette)
sns.set_context("talk")
palplot(palette)
stashfig("palette")

#%% [markdown]
# ## Run the layout on the 'sum' graph
#%%
layout = "kamada_kawai"


if layout == "spring":
    pos = nx.spring_layout(
        g, k=5 / np.sqrt(len(g)), weight=None, seed=88, iterations=200, threshold=1e-6
    )
else:
    pos = nx.kamada_kawai_layout(g, weight=None)


def layoutplot(
    g,
    pos,
    nodes,
    ax=None,
    figsize=(10, 10),
    weight_scale=1,
    adjust=True,
    log_weights=True,
    node_alpha=1,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    edgelist = g.edges()
    weights = np.array([g[u][v]["weight"] for u, v in edgelist])
    # weight transformations happen here, can be important
    if log_weights:
        weights = np.log(weights + 1)
    weights *= weight_scale

    # plot the actual layout
    nx.draw_networkx_nodes(g, pos, nodelist=nodes.index, node_color="white", zorder=-1)
    nx.draw_networkx_nodes(
        g, pos, nodelist=nodes.index, node_color=nodes["color"], alpha=node_alpha
    )
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=edgelist,
        nodelist=nodes.index,
        width=weights,
        zorder=-2,
        edge_color="lightgrey",
        alpha=1,  # connectionstyle="arc3,rad=0.2"
    )
    texts = []
    for node in nodes.index:
        node_pos = pos[node]
        split = node.split(" ")
        name = split[0] + "\n"
        name += node[len(split[0]) + 1 :]
        text = ax.text(
            node_pos[0],
            node_pos[1],
            name,
            ha="center",
            va="center",
            fontname="DejaVu Sans",
        )
        # text.set_bbox(dict(facecolor="white", alpha=0.3, edgecolor="white"))
        texts.append(text)

    ax.axis("off")

    if adjust:
        adjust_text(texts, expand_text=(1, 1), lim=100, avoid_self=False)

    return ax


sns.set_context("talk", font_scale=0.55)
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
layoutplot(g, pos, nodes, ax=ax, node_alpha=0.6, adjust=False)
stashfig("kavli")

#%%
def calculate_bounds(pos, pad=0.05):
    xy = np.stack(list(pos.values()), axis=1).T
    xmin = xy[:, 0].min()
    xmax = xy[:, 0].max()
    ymin = xy[:, 1].min()
    ymax = xy[:, 1].max()
    xpad = (xmax - xmin) * pad
    ypad = (ymax - ymin) * pad
    xlim = (xmin - xpad, xmax + xpad)
    ylim = (ymin - ypad, ymax + ypad)
    return xlim, ylim


core_mg = nx.subgraph(mg, nodes[nodes["core"]].index)
core_flat_g = flatten_muligraph(core_mg)
core_pos = nx.kamada_kawai_layout(core_flat_g, weight=None)
xlim, ylim = calculate_bounds(core_pos)

#%%


def extract_year_mgs(mg):
    year_mgs = {}
    for year in years:
        select_edges = []
        for (u, v, k, d) in mg.edges(data=True, keys=True):
            edge_year = d["start"]
            if edge_year <= year:
                select_edges.append((u, v, k))
        year_mg = nx.edge_subgraph(core_mg, select_edges)
        year_mgs[year] = year_mg
    return year_mgs


year_mgs = extract_year_mgs(core_mg)

for year in years:
    year_mg = year_mgs[year]
    year_g = flatten_muligraph(year_mg)
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    layoutplot(
        year_g,
        core_pos,
        nodes[nodes.index.isin(year_g.nodes)],
        ax=ax,
        node_alpha=0.6,
        adjust=False,
    )
    ax.set_title(year, fontsize=30)
    ax.set(xlim=xlim, ylim=ylim)
    stashfig(f"kavli-core-year-{year}")



#%%
# Questions
# - Who is the set of nodes to consider?
# - Some of them don't have joined dates, not sure if important
# - Some of the departments are wonky?
# - where to get the data for the pre, projected


# first graph would be all of the people who have joined
# people under the "joined" tab
# second graph to show

# joined in 2016 is start
# joined is only for the core group

# separate graph for
# could do this by the actual connections...

# edges will be cumulative
# one for each year

#

