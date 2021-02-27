#%%
import numpy as np
from graspologic.simulations import sbm

p1 = 0.7
p2 = 0.5
p3 = 0.1
p4 = 0.3
B1 = np.array([[p1, p3], [p3, p1]])  # affinity
B2 = np.array([[p1, p3], [p3, p2]])  # core-periphery
B3 = np.array([[p1, p2], [p2, p1]])  #
B4 = np.array([[p1, p4], [p4, p3]])

n = [50, 50]
A1, labels = sbm(n, B1, return_labels=True)
A2 = sbm(n, B2)
A3 = sbm(n, B3)
A4 = sbm(n, B4)


from graspologic.embed import AdjacencySpectralEmbed
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ortho_group
from graspologic.embed import selectSVD

sns.set_context("talk")

Vs = []
for i in range(10):
    As = [A1, A2, A3, A4]
    Xs = []
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i, A in enumerate(As):
        ase = AdjacencySpectralEmbed(n_components=2)
        X = ase.fit_transform(A)
        Q = ortho_group.rvs(X.shape[1])
        X = X @ Q
        Xs.append(X)
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, ax=axs[i], legend=False)
        # ax.set(xticks=[], yticks=[])
    plt.tight_layout()

    Z = np.concatenate(Xs, axis=1)
    # ase = AdjacencySpectralEmbed(n_components=2)
    # V = ase.fit_transform(Z)
    V, D, W = selectSVD(Z, n_components=2)
    Vs.append(V)

#%%
from graspologic.align import OrthogonalProcrustes

fig, axs = plt.subplots(2, 5, figsize=(20, 8))
axs = axs.flat
for i, V in enumerate(Vs):
    V_rot = OrthogonalProcrustes().fit_transform(V, Vs[0])
    # V_rot = V
    ax = axs[i]
    sns.scatterplot(x=V_rot[:, 0], y=V_rot[:, 1], hue=labels, ax=ax, legend=False)
    ax.set(xticks=[], yticks=[])

#%%
ZZt = Z @ Z.T
U, D, Ut_ = selectSVD(ZZt, n_components=2)


# %%
new_Xs = []
for i, X in enumerate(Xs):
    Q = ortho_group.rvs(X.shape[1])
    X = X @ Q
    new_Xs.append(X)

Z_w = np.concatenate(new_Xs, axis=1)
Z = np.concatenate(Xs, axis=1)
np.linalg.norm(Z_w @ Z_w.T - Z @ Z.T)

from graspologic.plot import heatmap

plt.figure()
sns.heatmap(Z_w)
plt.figure()
sns.heatmap(Z)
heatmap(Z_w @ Z_w.T)
heatmap(Z @ Z.T)

#%%
Q = ortho_group.rvs(X.shape[1])
XWYt = Xs[0] @ Q @ Xs[1].T
XYt = Xs[0] @ Xs[1].T
np.linalg.norm(XWYt - XYt)
#%%
block = [[Xs[0] @ Xs[0].T, Xs[0] @ Xs[1].T], [Xs[1] @ Xs[0].T, Xs[1] @ Xs[1].T]]
ZZt_mine = np.block(block)
heatmap(ZZt_mine)

#%%
ZZt_w = Z_w @ Z_w.T
ZZt_w.shape
# heatmap()

#%%
ZZt_mine = Xs[0] @ Xs[0].T + Xs[1] @ Xs[1].T + Xs[2] @ Xs[2].T + Xs[3] @ Xs[3].T
Z = np.concatenate(Xs, axis=1)
ZZt = Z @ Z.T
np.linalg.norm(ZZt - ZZt_mine)
