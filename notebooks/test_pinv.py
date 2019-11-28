# %% [markdown]
# #
import numpy as np

A = 10 * np.random.rand(10, 10)
A[:, 0] = 1

A = A * 1j
plt.figure()
# sns.heatmap(A)
b = np.random.rand(10, 1)
b = b * 1j
# plt.figure()
A_dag = np.linalg.pinv(A)
# sns.heatmap(A_dag)

A @ A_dag @ np.ones(10)
# plt.figure()
# sns.heatmap(A @ A_dag)
# %%
A_dag
plt.figure()
sns.heatmap(A_dag)
plt.figure()
sns.heatmap(b)

print(A_dag[0, :] - b.T)
# %%
b_hat = A_dag @ b

A @ b_hat - b_hat[0]

# %%
e = np.ones(10)
np.outer(e, e)


# %%
A = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 10]])
Adag = np.linalg.pinv(A)
Adag

A @ Adag

evals, evecs = np.linalg.eig(A @ Adag)
D = np.diag(evals)
U = evecs

print(U.T)
print(U.T @ np.ones(3))
print(D)
print(D @ U.T @ np.ones(3))
# D @ U.T @ np.ones(3)
# %%
