#%% [markdown]
# # Ranking teams by minimizing upsets via graph matching
# > Investigating the use of graph matching for ranking in a network
#
#- toc: true
#- badges: false
#- categories: [pedigo, graspologic, graph-match]
#- hide: false
#- search_exclude: false
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

sns.set_context("talk")

savefig_kws = dict(
    dpi=300, pad_inches=0.3, transparent=False, bbox_inches="tight", facecolor="white"
)

output_path = Path("sandbox/results/upset_ranking/")

# REF: https://www.sports-reference.com/cfb/years/2020.html
schedule = pd.read_csv("sandbox/data/ncaa_football_2020.csv")


def filt(string):
    return re.sub(r"\([0123456789]*\) ", "", string)


vec_filt = np.vectorize(filt)

schedule["Winner"] = vec_filt(schedule["Winner"])
schedule["Loser"] = vec_filt(schedule["Loser"])

unique_winners = np.unique(schedule["Winner"])
unique_losers = np.unique(schedule["Loser"])
teams = np.union1d(unique_winners, unique_losers)
n_teams = len(teams)


adjacency_df = pd.DataFrame(
    index=teams, columns=teams, data=np.zeros((n_teams, n_teams))
)

for idx, row in schedule.iterrows():
    winner = row["Winner"]
    loser = row["Loser"]
    adjacency_df.loc[winner, loser] += 1

remove_winless = False
n_wins = adjacency_df.sum(axis=1)
if remove_winless:
    teams = teams[n_wins > 0]
    n_wins = adjacency_df.sum(axis=1)
adjacency_df = adjacency_df.reindex(index=teams, columns=teams)

n_teams = len(teams)

ax, _ = adjplot(adjacency_df.values, plot_type="scattermap", sizes=(10, 10), marker="s")
ax.set_title("2020 NCAA Footbal Season Graph", fontsize=25)
ax.set(ylabel="Winning team")
ax.set(xlabel="Losing team")
plt.savefig(output_path / "unsorted_adjacency.png", **savefig_kws)

print(f"Number of teams {n_teams}")

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

