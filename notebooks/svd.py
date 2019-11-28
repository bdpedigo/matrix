# %% [markdown]
# #
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-white")
sns.set_palette("deep")
sns.set_context("talk")
figsize = (10, 10)

n = 100
true_mu = np.array([0, 0, 0])
print(true_mu)
true_cov = np.array([[2, 2, 0.5], [0, 1.5, 0.5], [0, 0, 1]])
true_cov = true_cov + true_cov.T
print(true_cov)

plt.figure(figsize=(10, 10))
sns.heatmap(true_cov, annot=True, square=True, cmap="Reds", cbar=False)


data = np.random.multivariate_normal(true_mu, true_cov, size=(n)).T

plt.figure()
sns.scatterplot(data[0], data[1])
plt.figure()
sns.scatterplot(data[0], data[2])
plt.figure()
sns.scatterplot(data[1], data[2])

#%%


def unitary_diagonalization(A):

    evals, U = np.linalg.eig(A)
    inds = np.argsort(evals)[::-1]
    evals = evals[inds]
    U = U[:, inds]
    D = np.diag(evals)
    return U, D


cov = (1 / n) * data @ data.T
U, D = unitary_diagonalization(cov)

np.testing.assert_almost_equal(U @ D @ U.T, cov)  # verify that the unitary diag. worked

plt.figure(figsize=figsize)
sns.heatmap(cov, annot=True, square=True, cmap="Reds", cbar=False)
plt.title("Estimated covariance")

plt.figure(figsize=figsize)
sns.heatmap(D, annot=True, square=True, cmap="Reds", cbar=False)
plt.title("Diagonalized estimated covariance")
# %%
R = np.zeros((2, 3))  # this is the row remover
R[0, 0] = 1  # keep the first dimension
R[1, 2] = 1  # and keep the third dimensions
plt.figure()
sns.heatmap(R, cmap="Reds", cbar=False)
plt.title("Row remover matrix")

last_ind = 100
C = np.zeros((n, last_ind))  # this is the column remover
inds = np.arange(0, last_ind)
C[inds, inds] = 1
plt.figure()
sns.heatmap(C, cmap="Reds", cbar=False)
plt.title("Column remover matrix")

subdata = R @ data @ C  # remove rows/cols via matrix multiplication

subdata_by_inds = data[np.ix_([0, 2], inds)]  # do the same via indexing

np.testing.assert_equal(subdata, subdata_by_inds)  # verify that this worked

subdata_cov = (1 / n) * subdata @ subdata.T
plt.figure(figsize=figsize)
sns.heatmap(subdata_cov, annot=True, square=True, cmap="Reds", cbar=False, vmin=0)
plt.title("Estimated covariance of subdata")

U, D = unitary_diagonalization(subdata_cov)

plt.figure(figsize=figsize)
sns.heatmap(D, annot=True, square=True, cmap="Reds", cbar=False, vmin=0)


# %%
R.T @ R

# %%
