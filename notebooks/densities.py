#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%% problem 58

n_samples = 100000

U = np.random.rand(n_samples)
X = np.sqrt(U)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.histplot(X, ax=ax, stat="density")
xs = np.linspace(0, 1, 1000)
pred_line = 2 * xs
sns.lineplot(x=xs, y=pred_line, ax=ax, zorder=2, color="red")

#%% problem 59
n_samples = 100000
U = np.random.rand(n_samples) * 2 - 1
X = U ** 2
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.histplot(X, ax=ax, stat="density", bins=np.linspace(0, 1, 1000))
xs = np.linspace(0, 1, 1000)
# pred_line = 1 / (xs ** 2)
pred_line = 0.5 * xs ** (-1 / 2)
sns.lineplot(x=xs, y=pred_line, ax=ax, zorder=2, color="red")
ax.set(ylim=(0, 10))

#%% ch 3 problem 36
n_samples = 100000

x = np.linspace(-1, 1, 1000)
alpha = 0.5
f = (1 + alpha * x) / 2

# t = np.random.uniform(-1, 1, size=n_samples)
M = np.max(f)


def density(x, alpha=0.5):
    return (1 + alpha * x) / 2


def proposal():
    t = np.random.uniform(-1, 1)
    y = np.random.uniform() * M
    return t, y


accepted = []
for i in range(n_samples):
    t, y = proposal()
    if y <= density(t):
        accepted.append(t)


fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.lineplot(x=x, y=f, ax=ax, color="red")
sns.histplot(accepted, stat="density", bins=np.linspace(-1, 1, 100))

#%% ch 3 problem 37
n_samples = 100000


def density(x):
    return 6 * (x ** 2) * ((1 - x) ** 2)


xs = np.linspace(-1, 1, 1000)
fs = np.array([density(x) for x in xs])
M = np.max(fs)


def proposal():
    t = np.random.uniform(-1, 1)
    y = np.random.uniform() * M
    return t, y


accepted = []
points = []
for i in range(n_samples):
    t, y = proposal()
    is_accepted = y <= density(t)
    if is_accepted:
        accepted.append(t)
    points.append({"t": t, "y": y, "accepted": is_accepted})


fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.lineplot(x=xs, y=fs * 0.16666, ax=ax, color="red")
M_line = np.ones(len(xs)) * M
# sns.lineplot(x=xs, y=M_line, ax=ax, color="forestgreen")
sns.histplot(accepted, stat="density", bins=np.linspace(-1, 1, 100), element="poly")

# %%

#%%
import pandas as pd

points = pd.DataFrame(points)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.scatterplot(data=points, x="t", y="y", hue="accepted", s=1, linewidth=0)
sns.lineplot(x=xs, y=fs, ax=ax, color="red")

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.kdeplot(accepted, ax=ax)


# %% monte carlo integration

from scipy.stats import norm


def density(x):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-(x ** 2) / 2)


a = 0
b = 2
n_samples = 1000000
support = np.random.uniform(a, b, n_samples)
points = density(support)
integral = (b - a) * 1 / len(support) * np.sum(points)
print(integral)


normal = norm(0, 1)
exact = normal.cdf(b) - normal.cdf(a)
print(exact)

#%%
