#%%
import numpy as np
import seaborn as sns

names = ["Danny", "Ben", "Nat"]

n = len(names)
d = 2

indices = np.arange(n * d)

no_loops = False
rowmin_d = False
colmin_d = False
i = 0
while not (no_loops and rowmin_d and colmin_d):
    pairs = np.random.permutation(indices)

    indices == pairs

    A = np.zeros((n, n))

    A[indices // d, pairs // d] = 1

    rowmin_d = A.sum(axis=1).min() == d
    colmin_d = A.sum(axis=0).min() == d
    no_loops = np.diag(A).max() == 0

    i += 1

sns.heatmap()
