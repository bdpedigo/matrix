#%%
import numpy as np

rng = np.random.default_rng()

n_samples = 1000
X = np.random.multivariate_normal([0, 0], np.eye(2), (n_samples))

A = np.array([[0.5, -0.5], [-0.5, 0.5]])
Y = (A @ X.T).T
np.corrcoef(Y[:, 0], Y[:, 1])

#%%
A = np.array([[0.5, 0.5], [-0.5, 0.5]])
A @ A.T