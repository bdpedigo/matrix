#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm

mu1 = 0
mu2 = 10
sigma1 = 1
sigma2 = 1
pi = 0.5

#%%
# dumb way of sampling just for exposition
np.random.seed(888)
n_samples = 100
X = np.empty(n_samples)
labels = np.empty(n_samples)
for i in range(n_samples):
    indicator = np.random.binomial(1, pi)
    if indicator:
        xi = np.random.normal(mu1, sigma1)
    else:
        xi = np.random.normal(mu2, sigma2)
    X[i] = xi
    labels[i] = int(indicator)

#%%

sns.set_context("talk")


def plot_likelihood(mu, sigma, ax, xmin=-4, xmax=14, samples=1000, **kwargs):
    xs = np.linspace(xmin, xmax, samples)
    ys = norm(mu, sigma).pdf(xs)
    sns.lineplot(x=xs, y=ys, ax=ax, **kwargs)


colors = sns.color_palette("deep", 4, desat=1)
palette = dict(zip([0, 1], colors))

mu1_hat = 0
mu2_hat = 6.5
sigma1_hat = 1
sigma2_hat = 1
pi_hat = 0.5

n_iter = 10
responsibilities = np.zeros(len(X))

for iteration in range(n_iter):
    # E step
    # compute responsibilities
    # this is just the posterior ratio of the different mixture components
    for i, xi in enumerate(X):
        component1_lik_prior = norm.pdf(xi, mu1_hat, sigma1_hat) * (1 - pi_hat)
        component2_lik_prior = norm.pdf(xi, mu2_hat, sigma2_hat) * pi_hat
        responsibilities[i] = component2_lik_prior / (
            component1_lik_prior + component2_lik_prior
        )
    # print(np.sum(responsibilities))
    # M step
    # compute paramaters to maximize likelihood given the responsibilities/data
    # for i, xi, responsibility in enumerate(zip(X, responsibilities)):
    mu1_hat = np.sum((1 - responsibilities) * X) / np.sum(1 - responsibilities)
    mu2_hat = np.sum(responsibilities * X) / np.sum(responsibilities)
    sigma1_hat = np.sum((1 - responsibilities) * (X - mu1_hat) ** 2) / np.sum(
        (1 - responsibilities)
    )
    sigma2_hat = np.sum(responsibilities * (X - mu2_hat) ** 2) / np.sum(
        responsibilities
    )
    pi_hat = np.sum(responsibilities) / len(X)
    # print(mu1_hat)
    # print(mu2_hat)
    # print(sigma1_hat)
    # print(sigma2_hat)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.rugplot(X, hue=labels, ax=ax, palette=palette)
    sns.histplot(
        x=X, hue=labels, bins=np.linspace(-2, 12, 30), stat="density", palette=palette
    )
    plot_likelihood(mu1_hat, sigma1_hat, ax, color=colors[2])
    plot_likelihood(mu2_hat, sigma2_hat, ax, color=colors[3])


# %%


def fit_gmm(X, mu1_hat, mu2_hat, sigma1_hat=1, sigma2_hat=1, pi_hat=0.5, n_iter=20):
    mu1_path = [mu1_hat]
    mu2_path = [mu2_hat]

    responsibilities = np.zeros(len(X))
    for iteration in range(n_iter):
        # E step
        # compute responsibilities
        # this is just the posterior ratio of the different mixture components
        for i, xi in enumerate(X):
            component1_lik_prior = norm.pdf(xi, mu1_hat, sigma1_hat) * (1 - pi_hat)
            component2_lik_prior = norm.pdf(xi, mu2_hat, sigma2_hat) * pi_hat
            responsibilities[i] = component2_lik_prior / (
                component1_lik_prior + component2_lik_prior
            )

        # M step
        # compute paramaters to maximize likelihood given the responsibilities/data
        # for i, xi, responsibility in enumerate(zip(X, responsibilities)):
        mu1_hat = np.sum((1 - responsibilities) * X) / np.sum(1 - responsibilities)
        mu2_hat = np.sum(responsibilities * X) / np.sum(responsibilities)
        sigma1_hat = np.sum((1 - responsibilities) * (X - mu1_hat) ** 2) / np.sum(
            (1 - responsibilities)
        )
        sigma2_hat = np.sum(responsibilities * (X - mu2_hat) ** 2) / np.sum(
            responsibilities
        )
        pi_hat = np.sum(responsibilities) / len(X)

        mu1_path.append(mu1_hat)
        mu2_path.append(mu2_hat)

    return mu1_path, mu2_path


mu_guesses = np.linspace(-4, 14, 10)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
for i, mu1_hat in enumerate(mu_guesses):
    for j, mu2_hat in enumerate(mu_guesses):
        print((10 * i + j) / 100)
        mu1_path, mu2_path = fit_gmm(X, mu1_hat, mu2_hat)
        sns.lineplot(x=mu1_path, y=mu2_path, color="")
