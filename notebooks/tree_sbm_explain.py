#%%
import time

import matplotlib.pyplot as plt
import numpy as np
from giskard.hierarchy import LeidenTree
from giskard.plot import plot_dendrogram, set_theme
from graspologic.plot import adjplot
from graspologic.simulations import sbm

t0 = time.time()
set_theme()


def stashfig(
    name,
    format="png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.1,
    transparent=False,
    facecolor="white",
    **kws,
):
    plt.savefig(
        "sandbox/results/tree_sbm_explain/" + name + f".{format}",
        format=format,
        dpi=dpi,
        bbox_inches=bbox_inches,
        transparent=transparent,
        pad_inches=pad_inches,
        facecolor=facecolor,
    )

#%%


