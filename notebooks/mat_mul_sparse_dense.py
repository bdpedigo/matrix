#%%
import numpy as np
from scipy.sparse import csr_matrix

n = 100
A = np.random.randint(0, 2, size=(n, n))
B = np.random.randint(0, 1000, size=(n, n))
C = np.random.randint(0, 2, size=(n, n))
A = csr_matrix(A)
C = csr_matrix(C)
print(A.count_nonzero())
A @ B @ C
