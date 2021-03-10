#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

n = 100
mu = 0
data = np.random.normal(loc=mu, size=n)


def superefficient(data, a=0.5):
    X_bar = np.mean(data)
    if np.abs(X_bar) >= n ** (-1 / 4):
        return X_bar
    else:
        return a * X_bar


def bic(data):
    X_bar = np.mean(data)
    if np.abs(X_bar) < np.sqrt(np.log(len(data)) / len(data)):
        return 0.0
    else:
        return X_bar


def basic(data):
    return np.mean(data)


from scipy.stats import norm


def aic(data):
    X_bar = np.mean(data)

    # model 1
    log_lik = norm.logpdf(data, loc=X_bar).sum()
    aic_1 = 2 * 1 - 2 * log_lik

    # model 0
    log_lik = norm.logpdf(data, loc=0).sum()
    aic_0 = -2 * log_lik

    if aic_0 <= aic_1:
        return 0.0
    else:
        return X_bar


ns = [100]
mus = np.linspace(0, 1.5, 50)
n_trials = 10000

rows = []

for i in range(n_trials):
    for n in ns:
        for mu in mus:
            data = np.random.normal(loc=mu, size=n)

            # hodges
            mu_hat = superefficient(data)
            ss = (mu_hat - mu) ** 2
            risk = ss * n
            row = {
                "i": i,
                "n": n,
                "mu": mu,
                "ss": ss,
                "risk": risk,
                "estimator": "superefficient",
            }
            rows.append(row)

            # bic
            mu_hat = bic(data)
            ss = (mu_hat - mu) ** 2
            risk = ss * n
            row = {"i": i, "n": n, "mu": mu, "ss": ss, "risk": risk, "estimator": "bic"}
            rows.append(row)

            # aic
            mu_hat = aic(data)
            ss = (mu_hat - mu) ** 2
            risk = ss * n
            row = {"i": i, "n": n, "mu": mu, "ss": ss, "risk": risk, "estimator": "aic"}
            rows.append(row)

            # no model selection
            mu_hat = basic(data)
            ss = (mu_hat - mu) ** 2
            risk = ss * n
            row = {
                "i": i,
                "n": n,
                "mu": mu,
                "ss": ss,
                "risk": risk,
                "estimator": "basic",
            }
            rows.append(row)

results = pd.DataFrame(rows)

#%%
sns.set_context("talk")
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.lineplot(data=results, y="risk", x="mu", hue="estimator", style="n")
plt.savefig("hodges_plot_sim")


# %%
