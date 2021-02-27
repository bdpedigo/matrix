#%%
import numpy as np

# A = np.array([[0, 1,2 3], [0, ]])
A = np.random.randint(0, high=100, size=(4, 4))
A

#%%
i, j = np.indices(A.shape)
#%%
i = i.ravel()
j = j.ravel()

#%%
print(A)
sort_inds = np.argsort(A, axis=1)

print(sort_inds)
# j[sort_inds]

#%%

values = A.ravel()

B = np.empty(A.shape)
B[, ] = values
B
A


#%%

X = np.random.randint(0, high=100, size=(5, 2))
print(X)
degrees = np.sqrt(np.array([1, 2, 3, 4, 5]))
Y = X * degrees[:, None]
print(Y)
dists = Y @ Y.T
print(dists)


#%%
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

rng = np.random.default_rng(seed=12345)
foo = rng.random((3000, 128))
euc = distance.cdist(foo, foo, metric=distance.euclidean)
nn = NearestNeighbors(n_neighbors=100, metric='precomputed').fit(euc)
distances, indices = nn.kneighbors(euc)
distances[100]
indices[100]

