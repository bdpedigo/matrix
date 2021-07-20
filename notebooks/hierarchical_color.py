#%%
import colorcet as cc
from colormath.color_diff import delta_e_cie2000
from matplotlib.colors import to_rgba
from colormath.color_objects import HSLColor, sRGBColor, LabColor, LCHabColor
from colormath.color_conversions import convert_color
from graspologic.utils import cartesian_product
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from graspologic.utils import is_almost_symmetric, symmetrize
import matplotlib.pyplot as plt
import pandas as pd
from graspologic.embed import ClassicalMDS


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
        "sandbox/results/hierarchical_color/" + name + f".{format}",
        format=format,
        dpi=dpi,
        bbox_inches=bbox_inches,
        transparent=transparent,
        pad_inches=pad_inches,
        facecolor=facecolor,
    )


#%%


def get_rgb(x):
    rgba = to_rgba(x)
    return rgba[:3]


colors = list(map(get_rgb, cc.glasbey_light))
color_objs = [sRGBColor(*rgb) for rgb in colors]
color_objs = [convert_color(x, LabColor) for x in color_objs]

color_pairs = cartesian_product(color_objs, color_objs)

color_dist_mat = np.empty((len(colors), len(colors)))
for i, color1 in enumerate(color_objs):
    for j, color2 in enumerate(color_objs):
        dist = delta_e_cie2000(color1, color2)
        color_dist_mat[i, j] = dist

print(is_almost_symmetric(color_dist_mat))
color_dist_mat = symmetrize(color_dist_mat)
#%%
Z = linkage(squareform(color_dist_mat), method="average")
sns.clustermap(
    color_dist_mat,
    row_colors=colors,
    col_colors=colors,
    row_linkage=Z,
    col_linkage=Z,
    xticklabels=False,
    yticklabels=False,
)
stashfig("clustermap")
# %%

cmds = ClassicalMDS(n_components=2, dissimilarity="precomputed")
color_embed = cmds.fit_transform(color_dist_mat)


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plot_df = pd.DataFrame(data=color_embed)
plot_df["labels"] = range(len(plot_df))
palette = dict(zip(range(len(plot_df)), colors))
sns.scatterplot(
    data=plot_df, x=0, y=1, palette=palette, hue="labels", ax=ax, legend=False, s=80
)
ax.axis("off")
stashfig("cmds")

#%%

# %%

n_points_per_level = 8
np.random.seed(8888)
points = [[0, 0]]
scale = 0.01
n_levels = 3
labels_by_level = []
points_by_level = []
for level in range(n_levels):
    next_points = []
    next_labels = []
    for i, point in enumerate(points):
        points_at_level = np.random.multivariate_normal(
            point, (scale ** level) * np.eye(2), size=n_points_per_level
        )
        next_points.append(points_at_level)
        next_labels.append(n_points_per_level * [i])
    next_points = np.concatenate(next_points)
    next_labels = np.concatenate(next_labels)
    points = next_points
    points_by_level.append(next_points)
    labels_by_level.append(next_labels)

for X, labels in zip(points_by_level, labels_by_level):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_df = pd.DataFrame(data=X)
    plot_df["labels"] = labels
    sns.scatterplot(ax=ax, data=plot_df, x=0, y=1, hue="labels")
    stashfig("labelplot")
# %%

n_steps = 50

l_range = np.linspace(30, 80, n_steps)
c_range = np.linspace(20, 80, n_steps)
h_range = np.linspace(0, 360, n_steps)

hcl_grid = cartesian_product(l_range, c_range, h_range)


def lch_to_rgb(l, c, h):
    lchab = LCHabColor(*(l, c, h))
    srgb = convert_color(lchab, sRGBColor)
    r = srgb.clamped_rgb_r
    g = srgb.clamped_rgb_g
    b = srgb.clamped_rgb_b
    return r, g, b


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=8, n_init=30, random_state=8888)
kmeans.fit(hcl_grid)

labels = kmeans.predict(hcl_grid)
centers = kmeans.cluster_centers_

dists_to_centers = kmeans.transform(hcl_grid)

min_dists = np.min(dists_to_centers, axis=1)
dists_to_centers[np.arange(len(labels)), labels] = np.inf
second_min_dists = np.min(dists_to_centers, axis=1)
dist_ratio = min_dists / second_min_dists

rgb_centers = []
for color in centers:
    r, g, b = lch_to_rgb(*color)
    rgb_centers.append((r, g, b))

sns.palplot(rgb_centers)
stashfig('palplot')

palette = {}
uni_labels = np.unique(labels)
for ul in uni_labels:

    kmeans = KMeans(n_clusters=8, n_init=10, random_state=8888)
    kmeans.fit(hcl_grid[(labels == ul) & (dist_ratio < 0.75)])
    centers = kmeans.cluster_centers_

    rgb_centers = []
    for i, color in enumerate(centers):
        r, g, b = lch_to_rgb(*color)
        rgb_centers.append((r, g, b))
        palette[(ul, i)] = (r, g, b)
    sns.palplot(rgb_centers)
    stashfig('palplot-next')
# %%
# for i in range(len(labels))
lvl1_labels = np.repeat(labels_by_level[1], n_points_per_level)
lvl2_labels = labels_by_level[2] % n_points_per_level
labels = [(a, b) for a, b in zip(lvl1_labels, lvl2_labels)]
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
plot_df = pd.DataFrame(data=X)
plot_df["labels"] = labels
sns.scatterplot(
    ax=ax, data=plot_df, x=0, y=1, hue="labels", palette=palette, legend=False
)
stashfig("scatter")
# labels = []
# for point in points_lvl_1:
#     points_lvl_2 = np.random.multivariate_normal(
#         point, scale * np.eye(2), size=n_points_per_level
#     )
#     labels.append([])

# plot_df = pd.DataFrame(data=points_lvl_2)
# plot_df["labels"] = labels
