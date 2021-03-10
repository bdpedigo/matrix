#%%
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
from sklearn.utils._testing import create_memmap_backed_data
from sklearn.cluster import AgglomerativeClustering

# code copied from `check_clustering` in sklearn/utils/estimator_checks.py
X, y = make_blobs(n_samples=100, random_state=30)
X, y = shuffle(X, y, random_state=7)
X = StandardScaler().fit_transform(X)
rng = np.random.RandomState(7)
X_noise = np.concatenate([X, rng.uniform(low=-3, high=3, size=(5, 2))])
X, y, X_noise = create_memmap_backed_data([X, y, X_noise])

import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.scatterplot(x=X[:, 0], y=X[:, 1], ax=ax)

# the params that triggered the error
# breaks for linkage='single', affinity "l1", "l2"
ag = AgglomerativeClustering(affinity="euclidean", linkage="single")
ag.fit(X)

#%%

len(np.unique(X, axis=0))
