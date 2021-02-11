#%%
import numpy as np

correct = np.array(
    [
        [0, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1, 1, 1, 1, 1],
    ],
    dtype=float,
)
correct[-1, 0] = np.nan

#%%
import pandas as pd

df = pd.DataFrame(data=correct)

#%%

n_per_week = np.arange(11, 2, -1)

#%%
non_cheaters = correct[:-1]
p_hat_per_week = np.mean(non_cheaters, axis=0)
alpha_per_week = p_hat_per_week * n_per_week
alpha = np.mean(alpha_per_week)

# %%

from scipy.stats import bernoulli
from scipy.special import binom

p_per_week = alpha / n_per_week


#%%

from sandbox.notebooks.poibin import PoiBin

n_weeks = len(p_per_week)
ks = np.arange(0, n_weeks + 1)
pmf = PoiBin(p_per_week).pmf(ks)

#%%

import seaborn as sns
import matplotlib.pyplot as plt

names = np.array(
    [
        "David",
        "Danny",
        "Nat",
        "Mackenzie",
        "Eileen",
        "Nicole",
        "Mia",
        "Cecelia",
        "Stefani",
        "Hanna",
        "Prabarna",
    ]
)
n_correct = np.nansum(correct, axis=1)

sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(x=ks, y=pmf, ax=ax)
sns.scatterplot(x=ks, y=pmf, ax=ax)

# ax.axvline(7, linestyle="--", color="darkred")
uni_correct = np.unique(n_correct)
for n in uni_correct:
    these_names = names[n_correct == n]
    if n != 2:
        xytext = (25, 25)
    else:
        xytext = (5, 45)

    ax.annotate(
        list(these_names),
        (n, pmf[np.where(ks == n)[0]]),
        xytext,
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", relpos=(0, 0), shrinkB=3),
    )

# ax.axvline(7, color='darkred')
ax.fill_between(ks[-3:], pmf[-3:], y2=-0.01, zorder=-1, alpha=0.6)
tail_prob = np.sum(pmf[-3:])
ax.annotate(
    f"Tail mass: {tail_prob:0.5f}",
    (9, -0.005),
    xytext=(40, 0),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", relpos=(0, 0.5), shrinkB=3),
)

ax.set(ylabel="Likelihood", xlabel="Number correct")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
