#%%

import datetime
import time
from abc import abstractmethod
from pathlib import Path

import anytree
import colorcet as cc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from anytree import NodeMixin
from giskard.plot import graphplot, set_theme, dendrogram_barplot
from graspologic.partition import leiden
from graspologic.plot import adjplot
from sklearn.base import BaseEstimator

t0 = time.time()
set_theme()

#%%
data = pd.DataFrame(index=range(100))
data["x"] = np.random.uniform(size=100)
data["labels_0"] = 50 * [0] + 50 * [1]
data["labels_1"] = 25 * [0] + 25 * [1] + 25 * [0] + 25 * [1]
data["labels_2"] = 12 * [0] + 13 * [1] + 75 * [np.nan]
data["hue"] = np.random.randint(low=0, high=10, size=len(data))
# data['hue_order'] =
#%%
dendrogram_barplot(
    data,
    group="labels_",
    group_order="x",
    hue_order="x",
    hue="hue",
    pad=5,
    orient="h",
    figsize=(10, 2),
)
