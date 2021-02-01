#%%

import numpy as np

from scipy.stats import multivariate_normal

n_samples = 1000
dim = 2
cov = np.eye(dim)
mvn = multivariate_normal(mean=np.zeros(dim), cov=cov)

X = mvn.rvs(size=n_samples)
X -= X.mean(axis=0)

estimated_cov = (1 / n_samples) * X.T @ X

X += np.random.uniform(-0.05, 0.05, size=X.shape)

perturbed_cov = (1 / n_samples) * X.T @ X

diff = estimated_cov - perturbed_cov

diff_spectral_norm = np.linalg.norm(diff, ord=2)

estimated_cov_evals = np.linalg.eigvals(estimated_cov)
perturbed_cov_evals = np.linalg.eigvals(perturbed_cov)
diff_evals = np.abs(perturbed_cov_evals - estimated_cov_evals)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

fig, ax = plt.subplots(1, 1, figsize=(5, 2))
ax.scatter(estimated_cov_evals, np.zeros_like(estimated_cov_evals))
ax.scatter(perturbed_cov_evals, np.zeros_like(perturbed_cov_evals))
for eigval in estimated_cov_evals:
    circle = Circle(
        (eigval, 0),
        radius=diff_spectral_norm,
        edgecolor="darkred",
        linewidth=1,
        linestyle=":",
        facecolor="none",
    )
    ax.add_patch(circle)

#%%

from graspologic.simulations import sample_edges, sbm
from graspologic.utils import cartprod
import seaborn as sns

n_per_comm = 50
B = np.array([[0.8, 0.1, 0.1], [0.1, 0.75, 0.05], [0.1, 0.05, 0.6]])
_, labels = sbm([n_per_comm, n_per_comm, n_per_comm], B, return_labels=True)
P = B[np.ix_(labels, labels)]
sns.heatmap(P)

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
true_eigvals = np.linalg.eigvalsh(P)
n_sims = 1000
all_estimated_eigvals = []
for i in range(n_sims):
    A = sample_edges(P, directed=False, loops=True)
    estimated_eigvals = np.linalg.eigvalsh(A)
    all_estimated_eigvals += list(estimated_eigvals)

sns.histplot((all_estimated_eigvals), ax=ax, stat='density')

for true_eigval in true_eigvals[::-1][:3]:
    ax.axvline(true_eigval, color="darkred")

#%% 

np.linalg.norm(P - A, ord=2)