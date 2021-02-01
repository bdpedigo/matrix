# %% [markdown]
# #
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

n_samples = 100
n_features = 3
X, y = make_blobs(
    n_samples=n_samples, n_features=n_features, centers=None, cluster_std=3
)
y = y.astype(int)
data_df = pd.DataFrame(data=np.concatenate((X, y[:, np.newaxis]), axis=-1))
data_df.rename(columns={3: "Label"}, inplace=True)

# %% [markdown]
# # First take
sns.scatterplot(
    data=data_df,
    x=0,
    y=1,
    hue="Label",
    palette=sns.color_palette("Set1", data_df.Label.nunique()),
)

# %% [markdown]
# # Second take - use sns.set_context
sns.set_context("talk")
plt.figure(figsize=(10, 10))
sns.scatterplot(
    data=data_df,
    x=0,
    y=1,
    hue="Label",
    palette=sns.color_palette("Set1", data_df.Label.nunique()),
)

# %% [markdown]
# # Third take - remove junk!

# not that the seaborn context is already active so we don't need again
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(
    data=data_df,
    x=0,
    y=1,
    hue="Label",
    palette=sns.color_palette("Set1", data_df.Label.nunique()),
    ax=ax,
)
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())


# %%

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

n_samples = 200
n_features = 2
n_noise_features = 98
n_sims = 20
test_size = 0.3


def fit_model(model, model_name, train_test_data):
    X_train, X_test, y_train, y_test = train_test_data
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    temp_dict = {
        "Accuracy": acc,
        "Classifier": model_name,
        "# of PCs": X_train.shape[1],
    }
    return temp_dict


models = (
    KNeighborsClassifier(),
    RandomForestClassifier(n_estimators=100),
    SVC(gamma="scale"),
)
model_names = ("KNN", "RF", "SVM")

pc_out_dicts = []
for sim in range(n_sims):
    print(sim)
    X_blob, y_blob = make_blobs(
        n_samples=n_samples, centers=n_features, cluster_std=1.0, shuffle=True
    )
    X_noise = np.random.normal(0, 1, size=(n_samples, n_noise_features))
    X_blob = np.concatenate((X_blob, X_noise), axis=-1)
    X_blob = StandardScaler().fit_transform(X_blob)
    blob_pcs = PCA(n_components=X_blob.shape[1]).fit_transform(X_blob)
    for i in range(1, X_blob.shape[1] + 1):
        X = blob_pcs[:, :i]
        y = y_blob
        train_test_data = train_test_split(X, y, test_size=test_size)
        for model, model_name in zip(models, model_names):
            temp_dict = fit_model(model, model_name, train_test_data)
            pc_out_dicts.append(temp_dict)

pc_results_df = pd.DataFrame(pc_out_dicts)

raw_out_dicts = []
for sim in range(n_sims):
    print(sim)
    X_blob, y_blob = make_blobs(
        n_samples=n_samples, centers=n_features, cluster_std=1.0, shuffle=True
    )
    X_noise = np.random.normal(0, 1, size=(n_samples, n_noise_features))
    X_blob = np.concatenate((X_blob, X_noise), axis=-1)
    X_blob = StandardScaler().fit_transform(X_blob)
    for i in range(1, X_blob.shape[1] + 1):
        X = X_blob[:, :i]
        y = y_blob
        train_test_data = train_test_split(X, y, test_size=test_size)
        for model, model_name in zip(models, model_names):
            temp_dict = fit_model(model, model_name, train_test_data)
            raw_out_dicts.append(temp_dict)

raw_results_df = pd.DataFrame(raw_out_dicts)
# %% [markdown]
# #
# raw_results_df.rename(columns={""})
sns.set_context("talk")
fig, ax = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
sns.lineplot(
    x="# of PCs", y="Accuracy", data=raw_results_df, hue="Classifier", ax=ax[0]
)
sns.lineplot(x="# of PCs", y="Accuracy", data=pc_results_df, hue="Classifier", ax=ax[1])
plt.tight_layout()
