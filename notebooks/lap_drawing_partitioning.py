#%% [markdown]
# # Graph laplacian

#%%
import numpy as np
import networkx as nx
from graspologic.simulations import sbm
from graspologic.plot import heatmap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def eig(A):
    evals, evecs = np.linalg.eig(A)
    sort_inds = np.argsort(evals)
    evals = evals[sort_inds]
    evecs = evecs[:, sort_inds]
    return evals, evecs


#%%
g = nx.davis_southern_women_graph()

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
ax = axs[0]
nx.draw_random(g, ax=ax)
ax.set(title="Random")

ax = axs[1]
nx.draw_spectral(g, ax=ax)
ax.set(title="Spectral")


nodelist = list(sorted(g.nodes()))
A = nx.to_numpy_array(g, nodelist=nodelist)
degrees = np.sum(A, axis=0)
D = np.diag(degrees)
D_inv = np.diag(1 / degrees)
L = D - A
L_gen = D_inv @ L
evals, evecs = np.linalg.eig(L_gen)
sort_inds = np.argsort(evals)
evals = evals[sort_inds]
evecs = evecs[:, sort_inds]

locations = evecs[:, 1:3]

pos = dict(zip(nodelist, locations))
plt.figure()
nx.draw(g, pos=pos)

#%%


B = np.array([[0.8, 0.05], [0.05, 0.8]])
A = sbm([10, 10], B)
heatmap(A)


degrees = np.sum(A, axis=0)
D = np.diag(degrees)
L = D - A
evals, evecs = eig(L)
fig = plt.figure()
sns.scatterplot(y=evals, x=np.arange(len(evals)))

#%%

rows = []
for p in np.linspace(0, 0.5, 20):
    for i in range(10):
        B = np.array([[0.8, p], [p, 0.8]])
        A = sbm([10, 10], B)
        degrees = np.sum(A, axis=0)
        D = np.diag(degrees)
        L = D - A
        evals, evecs = eig(L)
        fiedler = evals[1]
        rows.append({"p": p, "fiedler": fiedler})

results = pd.DataFrame(rows)
plt.figure()
sns.scatterplot(data=results, x="p", y="fiedler")
sns.lineplot(data=results, x="p", y="fiedler")

#%%
data_path = "sandbox/data/male_chem_A_self_undirected.csv"
meta_path = "sandbox/data/master_cells.csv"
cells_path = "sandbox/data/male_chem_self_cells.csv"
adj = pd.read_csv(data_path, header=None)
meta = pd.read_csv(meta_path, header=None, index_col=0)
cells = np.squeeze(pd.read_csv(cells_path, header=None).values)
meta = meta.reindex(cells)
A = adj.values

degrees = np.sum(A, axis=0)
D = np.diag(degrees)
L = D - A

evals, evecs = eig(L)

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
hue = meta[6]
sns.scatterplot(y=evecs[:, 6], x=np.arange(len(evecs[:, 6])), hue=hue, ax=ax)

#%%
is_pharynx = meta[6]

heatmap(A, inner_hier_labels=is_pharynx)

#%%
g = nx.from_numpy_array(A)
nx.draw_spring(g)
