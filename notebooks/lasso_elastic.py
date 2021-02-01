#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ortho_group
from sklearn.linear_model import ElasticNet, Lasso

n_dims = 40
U = ortho_group.rvs(n_dims)

n_informative = 2
x = np.zeros(n_dims)
support = np.random.choice(n_dims, size=n_informative)
x[support] = np.random.exponential(5, size=n_informative)

y = U @ x


sigma = 0.02
n_repeats = 2
U_observed = []
for i in range(n_repeats):
    U_observed.append(U + np.random.normal(0, sigma))
U_observed = np.concatenate(U_observed, axis=1)


support_stacked = np.concatenate((support, support))


colors = sns.color_palette("deep", 10)
sns.set_context("talk")


def set_linestyles(lines, color):
    (markerline, stemlines, baseline) = lines
    plt.setp(stemlines, color=color, linestyle=":")
    plt.setp(markerline, color=color)


x = np.concatenate((x / 2, x / 2))
xs = np.arange(len(x))

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
lines = ax.stem(xs, x, use_line_collection=True, label="True")
set_linestyles(lines, colors[0])

# Lasso
model = Lasso(alpha=0.001, fit_intercept=False).fit(U_observed, y)
x_hat = model.coef_
lines = ax.stem(xs, x_hat, use_line_collection=True, label="Lasso")
set_linestyles(lines, colors[1])

# ElasticNet
model = ElasticNet(alpha=0.001, l1_ratio=0.75, fit_intercept=False).fit(U_observed, y)
x_hat = model.coef_
lines = ax.stem(xs, x_hat, use_line_collection=True, label="Elastic")
set_linestyles(lines, colors[2])
ax.legend(bbox_to_anchor=(1, 1), loc="upper left")

