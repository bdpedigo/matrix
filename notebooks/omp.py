#%%
import numpy as np
from sklearn.datasets import make_sparse_coded_signal

# REF https://scikit-learn.org/stable/auto_examples/linear_model/plot_omp.html#sphx-glr-auto-examples-linear-model-plot-omp-py
n_components, n_features = 512, 100
n_nonzero_coefs = 17

# generate the data

# y = Xw
# |x|_0 = n_nonzero_coefs

y, X, w = make_sparse_coded_signal(
    n_samples=1,
    n_components=n_components,
    n_features=n_features,
    n_nonzero_coefs=n_nonzero_coefs,
    random_state=0,
)

# distort the clean signal
sigma = 0.05
y_noisy = y + sigma * np.random.randn(len(y))


#%% omp implementation


def omp(A, y, k):
    col_norms = np.linalg.norm(A, axis=0)
    A_tilde = A / col_norms[None, :]

    x_hat = np.zeros(A.shape[1])
    residual = y
    support = []
    for i in range(k):
        j = np.argmax(np.abs(A_tilde.T @ residual))
        support.append(j)
        A_sub = A_tilde[:, support]
        x_hat_sub = np.linalg.inv(A_sub.T @ A_sub) @ A_sub.T @ y
        x_hat[support] = x_hat_sub
        residual = y - A_tilde @ x_hat
    W = np.diag(1 / col_norms)
    x_hat = W @ x_hat
    return x_hat


x_hat = omp(X, y_noisy, n_nonzero_coefs)

#%%
import matplotlib.pyplot as plt


fig, axs = plt.subplots(2, 1, figsize=(8, 6))
idx = w.nonzero()[0]
ax = axs[0]
ax.set_xlim(0, 512)
ax.stem(idx, w[idx], use_line_collection=True)

ax = axs[1]
ax.set_xlim(0, 512)
support = x_hat.nonzero()[0]
ax.stem(support, x_hat[support], use_line_collection=True)


# %%
n_sims = 10
for n_nonzero_coefs in [5, 10, 20, 50, 100]:
    for i in range(n_sims):

        y, X, w = make_sparse_coded_signal(
            n_samples=1,
            n_components=n_components,
            n_features=n_features,
            n_nonzero_coefs=n_nonzero_coefs,
            random_state=0,
        )

        # distort the clean signal
        sigma = 0.05
        y_noisy = y + sigma * np.random.randn(len(y))

        x_hat = omp(X, y_noisy, n_nonzero_coefs)
        error = np.linalg.norm(w - x_hat)
        print(error)

#%% IHT


def hard_threshold(x, k):
    inds = np.argpartition(np.abs(x), -k)[-k:]
    x_thresh = np.zeros_like(x)
    x_thresh[inds] = x[inds]
    return x_thresh


def iterative_hard_thresholding(A, b, k, x0=None, n_iter=10):
    eta = np.linalg.norm(A, ord=2) ** 2  # largetst singular value
    if x0 is None:
        x = np.full(A.shape[1], np.mean(b))

    for i in range(n_iter):
        g = x - (1 / eta) * A.T @ (A @ x - b)
        x = hard_threshold(g, k)

    return x


x_hat = iterative_hard_thresholding(X, y_noisy, n_nonzero_coefs)


#%%
n_components, n_features = 512, 100
n_nonzero_coefs = 30

b, X, x = make_sparse_coded_signal(
    n_samples=1,
    n_components=n_components,
    n_features=n_features,
    n_nonzero_coefs=n_nonzero_coefs,
    random_state=0,
)

# distort the clean signal
sigma = 0.05
b_noisy = b + sigma * np.random.randn(len(b))


fig, axs = plt.subplots(3, 1, figsize=(8, 9))
idx = x.nonzero()[0]
ax = axs[0]
ax.set_xlim(0, 512)
ax.stem(idx, w[idx], use_line_collection=True)

ax = axs[1]
ax.set_xlim(0, 512)
x_hat = omp(X, b, n_nonzero_coefs)
support = x_hat.nonzero()[0]
ax.stem(support, x_hat[support], use_line_collection=True)


ax = axs[2]
ax.set_xlim(0, 512)
x_hat = iterative_hard_thresholding(X, b, n_nonzero_coefs, n_iter=1000)
support = x_hat.nonzero()[0]
ax.stem(support, x_hat[support], use_line_collection=True)

#%%
n_components, n_features = 2000, 100
n_nonzero_coefs = 30

b, X, x = make_sparse_coded_signal(
    n_samples=1,
    n_components=n_components,
    n_features=n_features,
    n_nonzero_coefs=n_nonzero_coefs,
    random_state=0,
)


A = X
n_iter = 1000
k = n_nonzero_coefs
n_init = 50
eta = np.linalg.norm(A, ord=2) ** 2  # largest singular value
# x = np.full(A.shape[1], np.mean(b))

import pandas as pd
import seaborn as sns

x_best = np.zeros(A.shape[1])
A = A / np.linalg.norm(A, axis=0)
rows = []
for init in range(n_init):
    x = np.zeros(A.shape[1])
    inds = np.random.choice(len(x), replace=True)
    x[inds] = 1

    for iteration in range(n_iter):
        g = x - (1 / eta) * A.T @ (A @ x - b)
        x = hard_threshold(g, k)
        score = np.linalg.norm(A @ x - b)
        rows.append({"init": init, "iteration": iteration, "score": score})

    if np.linalg.norm(A @ x - b) < np.linalg.norm(A @ x_best - b):
        x_best = x

progress = pd.DataFrame(rows)
sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.lineplot(data=progress, x="iteration", y="score", hue="init", ax=ax, palette="husl")
ax.set_yscale("log")
ax.get_legend().remove()


fig, axs = plt.subplots(3, 1, figsize=(8, 9))
idx = x.nonzero()[0]
ax = axs[0]
ax.set_xlim(0, n_components)
ax.stem(idx, x[idx], use_line_collection=True)

ax = axs[1]
ax.set_xlim(0, n_components)
x_hat = omp(X, b, n_nonzero_coefs)
support = x_hat.nonzero()[0]
ax.stem(support, x_hat[support], use_line_collection=True)


ax = axs[2]
ax.set_xlim(0, n_components)
x_hat = x_best
support = x_hat.nonzero()[0]
ax.stem(support, x_hat[support], use_line_collection=True)
