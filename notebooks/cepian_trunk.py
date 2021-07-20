#%%

import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from giskard.plot import soft_axis_off

sns.set_context("talk")
model_prior = np.ones(3)
model_prior /= model_prior.sum()

class_prior = 0.5
d = 2
n = 10
mu = np.ones(d)

model_A_cov = np.array([[4, 1], [1, 1]])
model_D_cov = np.array([[3, 0], [0, 3 / 2]])
model_C_cov = np.array([[2, 0], [0, 2]])


def make_data(n, model):
    if model == "A":
        cov = model_A_cov
    elif model == "C":
        cov = model_C_cov
    elif model == "D":
        cov = model_D_cov

    X = np.random.multivariate_normal(mu, cov, size=n)
    y = 2 * np.random.binomial(1, class_prior, size=n) - 1
    X = X * y[:, None]

    plot_df = pd.DataFrame(data=X, columns=["0", "1"])
    plot_df["y"] = y
    return X, y, plot_df


X, y, plot_df = make_data(n, "D")

palette = dict(zip([-1, 1], sns.color_palette("deep")))
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.scatterplot(data=plot_df, x="0", y="1", hue="y", ax=ax, palette=palette)
ax.set(xlabel="Dimension 1", ylabel="Dimension 2")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

#%%


def estimate_cov(X, y, mu, model):
    X = X.copy() * y[:, None]
    X_centered = X - mu[None, :]
    gram = X_centered.T @ X_centered
    if model == "A":
        cov = 1 / len(X) * gram
    elif model == "D":
        cov = 1 / len(X) * np.diag(np.diag(gram))
    elif model == "C":
        cov = 1 / len(X) * np.diag(np.repeat(np.mean(np.diag(gram)), X.shape[1]))
    return cov


ns = [100, 500, 1000, 2000, 5000, 10000, 20000]
models = ["A", "D", "C"]
model_covs = {"A": model_A_cov, "D": model_D_cov, "C": model_C_cov}
n_sims = 100
rows = []
for n in ns:
    for model in models:
        for i in range(n_sims):
            X, y, _ = make_data(n, model)
            cov_hat = estimate_cov(X, y, mu, model)
            diff = cov_hat - model_covs[model]
            norm_diff = np.linalg.norm(diff)
            row = {"n": n, "model": model, "i": i, "norm_diff": norm_diff}
            rows.append(row)
results = pd.DataFrame(rows)

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(data=results, x="n", y="norm_diff", hue="model")
ax.set_yscale("log")

#%%
# look at the decision boundary...


def estimate_precision(cov):
    return np.linalg.inv(cov)


def classify(X, precision):
    precision_row_sums = precision.sum(axis=1)
    logodds = X @ precision_row_sums
    y_hat = np.ones(len(logodds))
    y_hat[logodds < 0] = -1
    return y_hat


model = "A"
X, y, plot_df = make_data(n, model)
cov_hat = estimate_cov(X, y, mu, model)
prec_hat = estimate_precision(cov_hat)
y_hat = classify(X, prec_hat)
plot_df["y_hat"] = y_hat
# plot_df["correct"] = True

palette = dict(zip([-1, 1], sns.color_palette("deep")))
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
sns.scatterplot(
    data=plot_df,
    x="0",
    y="1",
    hue="y",
    ax=ax,
    palette=palette,
    s=4,
    alpha=0.3,
    linewidth=0,
)
ax.set(xlabel="Dimension 1", ylabel="Dimension 2")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

#%%


def score_predictions(y_true, y_pred):
    return (y_true == y_pred).mean()


from scipy.stats import multivariate_normal

model_param_counts = {"A": 3, "D": 2, "C": 1}


def select_model(X, y, model_estimated_params):
    X = X.copy() * y[:, None]
    best_bic = -np.inf
    best_model = None
    for model, cov_hat in model_estimated_params.items():
        log_lik = multivariate_normal(mean=mu, cov=cov_hat).logpdf(X).sum()
        k = model_param_counts[model]
        bic = 2 * log_lik - k * 2  # np.log(len(X))
        if bic > best_bic:
            best_model = model
            best_bic = bic
    return best_model


n_sims = 10000
n_train = 10
n_test = 10000
rows = []
for i in range(n_sims):
    true_model = np.random.choice(models, p=model_prior)
    X, y, _ = make_data(n_train, true_model)

    # estimate covariance for assuming each of the other models
    # TODO a model selection estimator
    X_test, y_test, _ = make_data(n_test, true_model)
    model_estimated_params = {}
    for model in models:
        cov_hat = estimate_cov(X, y, mu, model)
        prec_hat = estimate_precision(cov_hat)
        model_estimated_params[model] = cov_hat

        # TODO compute performance
        y_pred = classify(X_test, prec_hat)
        score = score_predictions(y_test, y_pred)
        row = {
            "score": score,
            "i": i,
            "true_model": true_model,
            "method": model,
            "chosen_model": model,
        }
        rows.append(row)

    selected_model = select_model(X, y, model_estimated_params)
    ind = np.argwhere(np.array(models)[::-1] == selected_model)[0][0] + 1
    row = rows[-ind]
    assert row["method"] == selected_model
    row = row.copy()
    row["method"] = "AIC"
    rows.append(row)

results = pd.DataFrame(rows)

#%%
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
pivot_results = results.pivot_table(
    values="score", index="true_model", columns="method", aggfunc="mean"
)
pivot_results = pivot_results.reindex(["A", "D", "C"])
mean_scores = results.groupby("method")["score"].mean()
mean_scores = mean_scores.sort_values(ascending=False)
mean_scores.name = "All"
pivot_results = pivot_results.reindex(columns=mean_scores.index)
pivot_results = pivot_results.append(mean_scores)
sns.heatmap(
    pivot_results,
    square=True,
    annot=True,
    cbar=False,
    cbar_kws=dict(shrink=0.6),
    fmt=".3g",
)
ax.set(ylabel="True model", xlabel="Estimation model")
ax.axhline(3, color="black", linewidth=1.5)
plt.setp(ax.get_yticklabels(), rotation=0)
savefig_kws = dict(
    dpi=300, pad_inches=0.3, transparent=False, bbox_inches="tight", facecolor="white"
)
plt.savefig("sandbox/results/cepian_trunk/acc_by_model.png", **savefig_kws)


# %%
