#%%
import numpy as np

A = np.array([[0, 1], [0, 0]])
np.linalg.eig(A)

#%%
dim = 3
x = np.random.normal(size=(dim))
x /= np.linalg.norm(x)
e1 = np.array([1, 0, 0], dtype=float)
e1 /= np.linalg.norm(e1)
w = e1 - x
w /= np.linalg.norm(w, ord=2)
H = np.eye(dim) - 2 * np.outer(w, w.T)

#%% schur deflation


def householder_difference(x, y):
    dim = len(x)
    w = x - y
    if np.linalg.norm(w) <= 1e-16:
        return np.eye(dim)
    w /= np.linalg.norm(w)
    H = np.eye(dim) - 2 * np.outer(w, w.T)
    return H


def direct_sum(A, B):
    if A.shape[0] == 0:
        return B
    return np.block(
        [
            [A, np.zeros((A.shape[0], B.shape[1]))],
            [np.zeros((B.shape[0], A.shape[1])), B],
        ]
    )


n = 4
A = np.random.normal(size=(n, n))
A = A + A.T
A = np.array([[2, 3], [1, 4]])
n=2
# A = symmetrize(A)
B = A
U = np.eye(n)
for i in range(n):
    print(i)
    eig_vals, eig_vecs = np.linalg.eig(B)
    eig_val = eig_vals[0]
    eig_vec = eig_vecs[:, 0]
    first_basis = np.zeros(len(B))
    first_basis[0] = 1
    # H is unitary
    # H has eig_vec as its first column
    H = householder_difference(first_basis, eig_vec)

    # plt.figure()
    # sns.heatmap(np.abs(H.conjugate().T @ B @ H))

    current_unitary = direct_sum(np.eye(i), H)

    # plt.figure()
    # sns.heatmap(np.abs(current_unitary))
    U = U @ current_unitary
    B = U.conjugate().T @ A @ U
    print(B[i, i])
    print(eig_val)
    # print(B)
    B = B[i + 1 :, i + 1 :]


import seaborn as sns

plt.figure()
sns.heatmap(np.abs(U.conjugate().T @ A @ U), annot=True)
