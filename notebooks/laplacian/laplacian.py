# %% [markdown]
# #
import numpy as np
import seaborn as sns
from graspy.simulations import sbm
from graspy.plot import heatmap
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sns.set_context("talk")


def eig(A):
    evals, evecs = np.linalg.eig(A)
    sort_inds = np.argsort(evals)
    evals = evals[sort_inds]
    evecs = evecs[:, sort_inds]
    return evals, evecs


def unnormalized_laplacian(A):
    D = np.diag(A.sum(axis=0))
    L = D - A
    return L


def clean_scatterplot(x, y, pad_proportion=0.1, **kwargs):
    plt.locator_params(nbins=3, tight=True)

    x = np.array(x)
    y = np.array(y)

    x_min = x.min()
    x_max = x.max()
    x_range = x_max - x_min
    y_min = y.min()
    y_max = y.max()
    y_range = y_max - y_min

    ax = sns.scatterplot(x, y, **kwargs)

    x_pad = x_range * pad_proportion / 2
    y_pad = y_range * pad_proportion / 2

    ax.set_xlim((x_min - x_pad, x_max + x_pad))
    ax.set_ylim((y_min - y_pad, y_max + y_pad))
    # ax.xaxis.set_major_formatter(
    #     ticker.FuncFormatter(lambda x, pos: "{:.2g}".format(x))
    # )
    x_ticklabels = ax.get_xticklabels()
    y_ticklabels = ax.get_yticklabels()
    # num_x
    # print((x_ticklabels[0], x_ticklabels[-1]))
    # print([i for i in x_ticklabels])
    # [t.set_visible(False) for t in x_ticklabels[1:-2]]
    # [t.set_visible(False) for t in y_ticklabels[1:-2]]

    # ax.set_xticklabels([])
    # ax.set_xticklabels((x_ticklabels[0], x_ticklabels[-1]))
    # ax.set_xticks((x_min, x_max))
    # ax.set_yticks((y_min, y_max))
    # ax.xaxis.set_major_formatter(plt.NullFormatter())

    # ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    # ax.yaxis.set_major_locator(plt.AutoLocator())
    # ax.yaxis.set_major_formatter(
    #     ticker.FuncFormatter(lambda x, pos: "{:.2g}".format(x))
    # )

    return ax


# %% [markdown]
# #

n_blocks = 4
n_per_comm = 50
n_verts = n_blocks * n_per_comm
comm_proportions = n_blocks * [n_per_comm]
low_p = 0.02
p_mat = np.array(
    [
        [0.5, low_p, low_p, low_p],
        [low_p, 0.4, low_p, low_p],
        [low_p, low_p, 0.55, low_p],
        [low_p, low_p, low_p, 0.45],
    ]
)
A, labels = sbm(comm_proportions, p_mat, return_labels=True)
heatmap(A, inner_hier_labels=labels, cbar=False)

# %% [markdown]
# # Compute some Laplacians


# The unnormalized graph Laplacian
L = unnormalized_laplacian(A)

heatmap(L, title="Unnormalized Graph Laplacian")

# L should be strictly PSD
evals, evecs = eig(L)


show_first = 20
plt.figure()
clean_scatterplot(range(show_first), evals[:show_first])
plt.xlabel("Eigenvalue index")
plt.ylabel("Magnitude")

# %% [markdown]
# We see that in the plot above, all of the eigenvalues are nonnegative and real

# %% [markdown]
# # Look at the smallest eigenvalue

print(f"First eigenvalue: {evals[0]}")

plt.figure()
# plt.plot(evecs[:, 0])
clean_scatterplot(range(n_verts), evecs[:, 0])
plt.title("First eigenvector")
plt.xlabel("Element")
plt.ylabel("Magnitude")
# plt.yticks([1])
# plt.xticks([0, n_verts])

# %% [markdown]
# # Look what happens for a disconnected graph

low_p = 0
p_mat = np.array(
    [
        [0.5, low_p, low_p, low_p],
        [low_p, 0.4, low_p, low_p],
        [low_p, low_p, 0.55, low_p],
        [low_p, low_p, low_p, 0.45],
    ]
)
A, labels = sbm(comm_proportions, p_mat, return_labels=True)
heatmap(A, inner_hier_labels=labels, cbar=False)

L = unnormalized_laplacian(A)
evals, evecs = eig(L)


plt.figure()
clean_scatterplot(range(show_first), evals[:show_first])
plt.xlabel("Eigenvalue index")
plt.ylabel("Magnitude")


print(f"First eigenvalue: {evals[0]}")

for i in range(n_blocks):
    plt.figure()
    clean_scatterplot(range(n_verts), evecs[:, i])
    plt.title(f"Eigenvector {i + 1}")
    plt.xlabel("Element")
    plt.ylabel("Magnitude")


# %%
from graspy.embed import LaplacianSpectralEmbed

lse = LaplacianSpectralEmbed(n_components=4)
vecs = lse.fit_transform(A)

plt.figure()
clean_scatterplot(range(n_verts), vecs[:, 0])
plt.figure()
clean_scatterplot(range(n_verts), vecs[:, 1])
plt.figure()
clean_scatterplot(range(n_verts), vecs[:, 2])
plt.figure()
clean_scatterplot(range(n_verts), vecs[:, 3])

#%%
degrees = np.sum(A, axis=1)
D = np.diag(degrees)
D_inv = np.diag(1 / degrees)
P = D_inv @ A
np.linalg.eig(P)

#%%
D_neg_half = np.diag(1 / np.sqrt(degrees))
DAD = D_neg_half @ A @ D_neg_half
np.linalg.eig(DAD)