#%%
import graspologic

import matplotlib.pyplot as plt
import numpy as np
from giskard.plot import set_theme

from graspologic.simulations import sbm
from graspologic.embed import OmnibusEmbed
import seaborn as sns
import pandas as pd


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
        "sandbox/results/omni_demonstrate/" + name + f".{format}",
        format=format,
        dpi=dpi,
        bbox_inches=bbox_inches,
        transparent=transparent,
        pad_inches=pad_inches,
        facecolor=facecolor,
    )


set_theme()

n = [15, 15]
P1 = [[0.3, 0.1], [0.1, 0.7]]
P2 = [[0.3, 0.1], [0.1, 0.3]]

np.random.seed(8)
G1 = sbm(n, P1)
G2 = sbm(n, P2)

embedder = OmnibusEmbed(n_components=2)
Zhat = embedder.fit_transform([G1, G2])

print(Zhat.shape)

Xhat1 = Zhat[0]
Xhat2 = Zhat[1]
Xhat_full = np.concatenate((Xhat1, Xhat2), axis=0)

colors = sns.color_palette("deep")

# Plot the points
fig, ax = plt.subplots(figsize=(8, 8))

labels = len(G1) * [r"Graph $t_1$"] + len(G2) * [r"Graph $t_2$"]

plot_df = pd.DataFrame(data=Xhat_full, columns=["0", "1"])
plot_df["labels"] = labels
# ax.scatter(Xhat1[:25, 0], Xhat1[:25, 1], marker="s", c=colors[0], label="Graph 1, Block 1")
# ax.scatter(Xhat1[25:, 0], Xhat1[25:, 1], marker="o", c=colors[0], label="Graph 1, Block 2")
# ax.scatter(Xhat2[:25, 0], Xhat2[:25, 1], marker="s", c=colors[1], label="Graph 2, Block 1")
# ax.scatter(Xhat2[25:, 0], Xhat2[25:, 1], marker="o", c=colors[1], label="Graph 2, Block 2")
# ax.legend()

sns.scatterplot(data=plot_df, x="0", y="1", hue="labels", s=100)
ax.set(
    xlabel="Embedding dimension 1", ylabel="Embedding dimension 2", xticks=[], yticks=[]
)
sns.move_legend(ax, "upper right", title=None)

# Plot lines between matched pairs of points
for i in range(sum(n)):
    ax.plot(
        [Xhat1[i, 0], Xhat2[i, 0]],
        [Xhat1[i, 1], Xhat2[i, 1]],
        "black",
        alpha=0.15,
        zorder=-1,
    )


i = 20
mean_x = np.mean([Xhat1[i, 0], Xhat2[i, 0]])
mean_y = np.mean([Xhat1[i, 1], Xhat2[i, 1]])
text = r"$d(X(t)_i, X(t')_i)$"
text = "Node's distance\nbetween timepoints"
ax.annotate(
    text,
    (mean_x, mean_y),
    xytext=(-10, 80),
    textcoords="offset points",
    arrowprops=dict(arrowstyle='->'),
    # arrowprops=dict(head_length=0.5,head_width=0.5,tail_width=0.2),
)

stashfig("omni-demonstrate")
# _ = ax.set_title("Latent Positions from Omnibus Embedding", fontsize=20)
