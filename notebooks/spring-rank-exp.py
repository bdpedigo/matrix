#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

s1 = np.linspace(-1, 1, 50)[::-1]
s2 = np.linspace(-1, 1, 50)[::-1]
beta = 10
H = (s1[:, None] - s1[None, :] - 1) ** 2
c = 1
P = c * np.exp(-beta / 2 * H)


sns.set_context("talk")
fig, axs = plt.subplots(
    2, 1, figsize=(6, 8), gridspec_kw=dict(height_ratios=[0.7, 0.3]), sharex=True
)
ax = axs[0]
sns.heatmap(P, cmap="RdBu_r", center=0, ax=ax, square=True, cbar_kws=dict(shrink=0.7))
ax.set(xticks=[], yticks=[])

ax = axs[1]
sns.scatterplot(x=np.arange(len(s1)), y=s1, ax=ax)

plt.figure()
sns.scatterplot(x=np.arange(len(s1)), y=P[0, :])
