#%%
import numpy as np
import seaborn as sns

p = 0.7
q = 0.2
B = np.array([[p ** 2 + q ** 2, 2 * p * q], [2 * p * q, p ** 2 + q ** 2]])

from graspologic.embed import AdjacencySpectralEmbed

# X, D, Y = selectSVD(B, n_components=1)
plt.figure()
sns.heatmap(B, square=True, annot=True)

X = AdjacencySpectralEmbed(n_components=1, diag_aug=False).fit_transform(B)

B_low_rank = X @ X.T

plt.figure()
sns.heatmap(B_low_rank, square=True, annot=True)
