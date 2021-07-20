#%%
import os
from pathlib import Path

import bibtexparser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import set_theme
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image


set_theme()


# Filepaths
tex_loc = "sandbox/data/connectome_time/bib/bib.tex"
image_dir = Path("sandbox/data/connectome_time/images")

# Load the bib data
with open(tex_loc) as f:
    bib_database = bibtexparser.load(f)
df = pd.DataFrame(bib_database.entries)

# Change some data types
df["n_nodes"] = df["n_nodes"].astype(int)
df["n_edges"] = df["n_edges"].fillna("0").astype(int)
df["year"] = df["year"].astype(int)

df
#%%
def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords="data", frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


figsize = (12, 6)
x_key = "year"
y_key = "n_nodes"

# padding the images away from the points
x_pad = 2
y_pad = 2000

fig, ax = plt.subplots(1, 1, figsize=figsize)
sns.scatterplot(data=df, x=x_key, y=y_key, color="darkorange")

for idx, row in df.iterrows():
    key = row["ID"]
    image_loc = image_dir / f"{key}.png"
    if os.path.isfile(image_loc):
        image = Image.open(image_loc)
        x = row[x_key] + x_pad
        y = row[y_key] + y_pad
        imscatter(x, y, image_loc, ax=ax, zoom=0.2)

    # TODO a description of the data is probably more helpful than this
    first_author_last_name = row["author"].split(",")[0]
    year = row["year"]

    short_name = f"{first_author_last_name} et al. {year}"
    ax.text(
        row[x_key] + 0.5,
        row[y_key] - 500,
        short_name,
        zorder=100,
        ha="left",
        va="top",
        fontsize="xx-small",
    )

ax.set(ylabel="# of nodes", xlabel="Year")
