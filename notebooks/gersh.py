# %% [markdown]
# #
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


def calculate_gersh(A):
    centers = np.diag(A)
    rads = np.sum(np.abs(A), axis=1) - np.abs(centers)
    centers = np.stack((np.real(centers), np.imag(centers)), axis=1)
    return centers, rads


def update_gersh(eps):
    B = D + eps * N
    centers, rads = calculate_gersh(B)


def get_circles(centers, rads):
    circles = []
    pal = sns.color_palette("deep", len(centers))
    for i, (c, r) in enumerate(zip(centers, rads)):
        circles.append(
            Circle(
                c,
                r,
                zorder=0,
                edgecolor="k",
                linewidth=1,
                # linestyle=":",
                facecolor=pal[i],
                alpha=0.3,
            )
        )
    return circles


def get_evals(centers):
    circles = []
    pal = sns.color_palette("deep", len(centers))
    r = 1
    for i, c in enumerate(centers):
        circles.append(
            Circle(
                c,
                r,
                zorder=0,
                edgecolor="k",
                linewidth=1,
                # linestyle=":",
                facecolor=pal[i],
                alpha=0.3,
            )
        )
    return circles


A = np.array([[0, -1, 1], [1, 0, 1], [1, 1, 1]])
A = A + np.ones((3, 3))
A = np.array([[0.5 + 5j, -1, 10j], [1, -0.05, 12j], [1j, 2 + 3j, 0]])
A = A + 10 * np.random.rand(3, 3)

evals, evecs = np.linalg.eig(A)


D = np.diag(np.diag(A))
N = A - D


sns.set_context("talk", font_scale=1)


D = np.diag(np.diag(A))
N = A - D

# set the bounds for the plot
full_centers, full_rads = calculate_gersh(A)
max_centers = full_centers + full_rads[:, np.newaxis]
min_centers = full_centers - full_rads[:, np.newaxis]

min_x, min_y = np.min(min_centers, axis=0)
max_x, max_y = np.max(max_centers, axis=0)
pad = 3

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(xlim=[min_x - pad, max_x + pad], ylim=[min_y - pad, max_y + pad])
scatter, = ax.plot(
    [], [], lw=3, linestyle="none", marker=".", c="r", label="Eigenvalue"
)
ax.set_xlabel("Re")
ax.set_ylabel("Im")
circ = plt.Circle((0.5, 0), 0.3)

plt.legend(
    [scatter, circ],
    [r"$\lambda$", "G-disc"],
    bbox_to_anchor=(1.03, 1),
    loc=2,
    borderaxespad=0.0,
)

# ax.legend()
patches = []


def init():
    scatter.set_data([], [])
    ax.set_xlim([min_x - pad, max_x + pad])
    ax.set_ylim([min_y - pad, max_y + pad])
    return scatter, []


def animate(i):
    B = D + i * N
    evals, evecs = np.linalg.eig(B)
    x = np.real(evals)
    y = np.imag(evals)
    scatter.set_data(x, y)
    for p in patches:
        print(type(p))
        p.remove()
    centers, rads = calculate_gersh(B)
    new_patches = get_circles(centers, rads)
    patches.clear()
    for p in new_patches:
        ax.add_patch(p)
        patches.append(p)
    return scatter, patches


anim = FuncAnimation(
    fig, animate, init_func=init, frames=np.linspace(0, 1, 100), interval=50
)
plt.tight_layout()
plt.draw()
plt.show()


# anim.save("coil.gif", writer="imagemagick")

# anim.save("sine_wave.html", writer="imagemagick")

