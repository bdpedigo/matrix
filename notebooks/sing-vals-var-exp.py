#%%
import numpy as np 

n = 10

A = np.random.normal(size=(n,n))

_, sing_vals, _ = np.linalg.svd(A)

sing_vals

#%%
# %%
from sklearn.decomposition import PCA

A = A - np.mean(A, axis=0)[None, :]
pca = PCA(n_components=n)
pca.fit_transform(A)
print('pca explained variance')
pca.explained_variance_ratio_

#%%
_, sing_vals, _ = np.linalg.svd(A)

print('sing val ratio')
sing_vals**2 / np.linalg.norm(A)**2

#%%
u, s, vh = np.linalg.svd(A)

rank1 = s[0]*np.outer(u[:, 0], vh[0, :])

print(1 - np.linalg.norm(A - rank1)**2 / np.linalg.norm(A)**2)

print(s[0]**2 / np.linalg.norm(A)**2)