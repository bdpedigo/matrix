#%% [markdown]
# # Ranking teams by minimizing upsets via graph matching
# > Investigating the use of graph matching for ranking in a network
#
# - toc: true
# - badges: false
# - categories: [pedigo, graspologic, graph-match]
# - hide: false
# - search_exclude: false
#%%[markdown]
# ## Constructing the graph
# Here I construct a network based on the games played during the 2020 NCAA football season.
# The nodes in the network represent teams and the directed edges represent a victory
# when team $i$ played team $j$.
#%%
import re
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from graspologic.match import GraphMatch
from graspologic.plot import adjplot
from graspologic.utils import remove_loops

sns.set_context("talk")

savefig_kws = dict(
    dpi=300, pad_inches=0.3, transparent=False, bbox_inches="tight", facecolor="white"
)

output_path = Path("sandbox/results/epl_ranking/")
league = "LALIGA1"
data_dir = Path("sandbox/data/sports/soccer")
start_year = 0
seasons = []
for year in range(start_year, 21):
    print(year)
    filename = f"{league}_{year}_{year+1}.csv"
    season_df = pd.read_csv(
        data_dir / filename, engine="python", usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8]
    )
    season_df["season"] = f"{year}/{year+1}"
    seasons.append(season_df)
games = pd.concat(seasons, ignore_index=True)

edges = []
games["source"] = "none"
games["target"] = "none"
for idx, row in games.iterrows():
    if row["FTHG"] > row["FTAG"]:
        source = row["HomeTeam"]
        target = row["AwayTeam"]
        weight = 1
    elif row["FTHG"] < row["FTAG"]:
        target = row["HomeTeam"]
        source = row["AwayTeam"]
        weight = 1
    else:
        target = row["HomeTeam"]
        source = row["AwayTeam"]
        weight = 1 / 2
        edges.append(
            {
                "source": source,
                "target": target,
                "season": row["season"],
                "weight": weight,
            }
        )
        source = row["HomeTeam"]
        target = row["AwayTeam"]
        weight = 1 / 2
    # ignore ties for now
    edges.append(
        {"source": source, "target": target, "season": row["season"], "weight": weight}
    )
edgelist = pd.DataFrame(edges)

nodelist = list(edgelist["source"].unique()) + list(edgelist["target"].unique())
nodelist = np.unique(nodelist)
#%%


def signal_flow(A):
    """Implementation of the signal flow metric from Varshney et al 2011

    Parameters
    ----------
    A : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    A = A.copy()
    A = remove_loops(A)
    W = (A + A.T) / 2

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    b = np.sum(W * np.sign(A - A.T), axis=1)
    L_pinv = np.linalg.pinv(L)
    z = L_pinv @ b

    return z


def rank_signal_flow(A):
    sf = signal_flow(A)
    perm_inds = np.argsort(-sf)
    return perm_inds


def rank_graph_match_flow(A, n_init=100, max_iter=50, eps=1e-5, **kwargs):
    n = len(A)
    initial_perm = rank_signal_flow(A)
    init = np.eye(n)[initial_perm]
    match_mat = np.zeros((n, n))
    triu_inds = np.triu_indices(n, k=1)
    match_mat[triu_inds] = 1
    gm = GraphMatch(
        n_init=n_init, max_iter=max_iter, init="barycenter", eps=eps, **kwargs
    )
    perm_inds = gm.fit_predict(match_mat, A)
    return perm_inds


def calculate_p_upper(A):
    A = remove_loops(A)
    n = len(A)
    triu_inds = np.triu_indices(n, k=1)
    upper_triu_sum = A[triu_inds].sum()
    total_sum = A.sum()
    upper_triu_p = upper_triu_sum / total_sum
    return upper_triu_p


import networkx as nx

rankings = []
ranking_stats = []
for season, season_edges in edgelist.groupby("season"):
    g = nx.from_pandas_edgelist(
        season_edges, edge_attr="weight", create_using=nx.MultiDiGraph
    )
    season_nodes = np.intersect1d(nodelist, g.nodes)
    adj = nx.to_numpy_array(g, nodelist=season_nodes)
    perm_inds = rank_graph_match_flow(adj, n_init=10)
    p_upper = calculate_p_upper(adj[np.ix_(perm_inds, perm_inds)])
    rankings.append(
        pd.Series(
            data=np.arange(len(season_nodes)),
            name=season,
            index=season_nodes[perm_inds],
        )
    )
    ranking_stats.append(
        {
            "p_upper": p_upper,
            "season": season,
            "season_start": int(season.split("/")[0]),
        }
    )
ranking_stats = pd.DataFrame(ranking_stats)
rankings = pd.DataFrame(rankings).T
rankings.index.name = "team"
rankings["mean"] = rankings.fillna(30).mean(axis=1)
rankings = rankings.sort_values("mean")
rankings
#%%
from graspologic.plot import heatmap

heatmap(adj)
heatmap(adj[np.ix_(perm_inds, perm_inds)])
season_nodes[perm_inds]

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

sns.lineplot(data=ranking_stats, x="season_start", y="p_upper")

#%%
# rankings = rankings.fillna(2)
import colorcet as cc

fig, ax = plt.subplots(1, 1, figsize=(20, 10))
pd.plotting.parallel_coordinates(
    rankings.fillna(30).reset_index().drop("mean", axis=1),
    class_column="team",
    ax=ax,
    color=cc.glasbey_light,
)
ax.get_legend().remove()
# ax.invert_yaxis()
ax.set(xlabel="Season", ylim=(21, -1), ylabel="Ranking", title=f"{league}")
ax.legend(bbox_to_anchor=(1, 1), loc="upper left", ncol=2)
plt.savefig(output_path / f"{league}_ranking.png", **savefig_kws)

# %%

#%% [markdown]
# ## Matching to a flat upper triangular matrix
# Under a given sorting (permutation) of the adjacency matrix, any game (edge) that is
# an upset will fall in the lower triangle, because a lower-ranked team beat a higher-ranked
# team. We can therefore create a ranking by graph matching the adjacency matrix to a
# flat upper triangular matrix, thereby inducing a sorting/ranking that minimizes the
# number of upsets.
# %%
adj = adjacency_df.values

# constructing the match matrix
match_mat = np.zeros_like(adj)
triu_inds = np.triu_indices(len(match_mat), k=1)
match_mat[triu_inds] = 1

# running graph matching
np.random.seed(8888)
gm = GraphMatch(n_init=500, max_iter=150, eps=1e-6)
gm.fit(match_mat, adj)
perm_inds = gm.perm_inds_

adj_matched = adj[perm_inds][:, perm_inds]
upsets = adj_matched[triu_inds[::-1]].sum()
n_games = adj_matched.sum()

print(f"Number of games: {n_games}")
print(f"Number of non-upsets (graph matching score): {gm.score_}")
print(f"Number of upsets: {upsets}")
print(f"Upset ratio: {upsets/n_games}")

print()
print("Ranking:")
print(teams[perm_inds])

#%% [markdown]
# ## Plotting the matched (ranked) graph
#%%
ax, _ = adjplot(adj_matched, plot_type="scattermap", sizes=(10, 10), marker="s")
ax.plot([0, n_teams], [0, n_teams], linewidth=1, color="black", linestyle="-")
ylabel = r"$\leftarrow$ Ranked low         "
ylabel += "Winning team           "
ylabel += r"Ranked high $\rightarrow$"
ax.set_ylabel(ylabel, fontsize="large")
ax.set(xlabel="Losing team")
ax.set_title("2020 NCAA Footbal Season Graph", fontsize=25)
ax.fill_between(
    [0, n_teams],
    [0, n_teams],
    [n_teams, n_teams],
    zorder=-1,
    alpha=0.4,
    color="lightgrey",
)
ax.text(n_teams / 4, 3 / 4 * n_teams, "Upsets")

plt.savefig(output_path / "ranked_adjacency.png", **savefig_kws)

