#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dim = 10
alpha = 0.2
n_samples = 100
mean = np.zeros(dim)
cov = alpha * np.eye(dim) + (1 - alpha) * np.ones((dim, dim))
X = np.random.multivariate_normal(mean, cov, size=(n_samples))
X = X @ np.diag(np.arange(1, dim + 1))

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], ax=ax)

#%%
X -= X.mean(axis=0)
estimated_cov = (1 / n_samples) * X.T @ X


# %%

A = estimated_cov


eigvals, eigvecs = np.linalg.eig(A)
sort_inds = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, sort_inds]
true_principal_eigvec = eigvecs[:, 0]

x = np.random.rand(dim)
errors = []
n_iter = 15
for i in range(n_iter):
    x = A @ x
    x /= np.linalg.norm(x)
    err = np.linalg.norm(x - true_principal_eigvec)
    errors.append(err)
    print(err)

eigval = x.T @ A @ x / (x.T @ x)

sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(errors)
ax.set_yscale("log")

#%%
B = A - np.outer(x, x.T) * eigval

# eigvals, eigvecs = np.linalg.eig(B)
# sort_inds = np.argsort(eigvals)[::-1]
# eigvecs = eigvecs[:, sort_inds]
true_principal_eigvec = eigvecs[:, 1]

y = np.random.rand(dim)
errors = []
n_iter = 100
for i in range(n_iter):
    y = B @ y
    y /= np.linalg.norm(y)
    err = np.linalg.norm(y - true_principal_eigvec)
    errors.append(err)
    print(err)


# %%
