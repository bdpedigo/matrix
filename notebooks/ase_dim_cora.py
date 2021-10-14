#%%
import graspologic as gl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import stellargraph as sg
from sklearn import model_selection
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

dataset = sg.datasets.Cora()
G, node_subjects = dataset.load()


g = G.to_networkx()

index = node_subjects.index

adj = nx.to_numpy_array(g, nodelist=index, weight="weight")

# note that this technically uses the test data in the
# ranking. we could do OOS PTR but haven't implemented it yet
# adj = gl.utils.pass_to_ranks(adj)

inds = np.arange(len(node_subjects))


models = [
    KNeighborsClassifier(),
    LogisticRegression(class_weight="balanced"),
    SVC(class_weight="balanced"),
    RandomForestClassifier(class_weight="balanced"),
    DummyClassifier(),
]
n_components_range = np.geomspace(2, 2 ** 10, 10, dtype="int")
n_splits = 20
rows = []
for split in range(n_splits):
    print(f"Split: {split+1}/{n_splits}")
    train_inds, test_inds = model_selection.train_test_split(
        inds, train_size=0.9, stratify=node_subjects
    )

    y_train = node_subjects.iloc[train_inds]
    y_test = node_subjects.iloc[test_inds]

    train_adj, oos_edges = gl.utils.remove_vertices(
        adj, indices=test_inds, return_removed=True
    )
    ase = gl.embed.AdjacencySpectralEmbed(
        n_components=n_components_range[-1], check_lcc=False
    )
    X_train = ase.fit_transform(train_adj)

    X_test = ase.transform(oos_edges)
    for n_components in n_components_range:
        for model in models:
            model.fit(X_train[:, :n_components], y_train)
            y_pred = model.predict(X_test[:, :n_components])

            acc = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average="macro")
            row = {
                "split": split,
                "n_components": n_components,
                "accuracy": acc,
                "model": model.__class__.__name__,
                "macro_f1": macro_f1,
            }
            rows.append(row)

results = pd.DataFrame(rows)

#%%
from giskard.plot import set_theme

set_theme()
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
sns.lineplot(data=results, x="n_components", y="accuracy", hue="model")
sns.move_legend(ax, "lower left", bbox_to_anchor=(0, 1), title=None, ncol=2)
ax.set_xlabel("# of dimensions")
ax.set_ylabel("Accuracy")
ax.set(xscale="log")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
sns.lineplot(data=results, x="n_components", y="macro_f1", hue="model")
sns.move_legend(ax, "lower left", bbox_to_anchor=(0, 1), title=None, ncol=2)
ax.set_xlabel("# of dimensions")
ax.set_ylabel("Macro F1")
ax.set(xscale="log")
