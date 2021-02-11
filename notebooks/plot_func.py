#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

xs = np.linspace(-10, 10, 100)
ys = np.linspace(-10, 10, 100)


X, Y = np.meshgrid(xs, ys)
a = 10


def f(x, y):
    return x ** 2 - x * (1 - x) / a + y ** 2 - y * (1 - y) / a


Z = f(X, Y)


fig = plt.figure(figsize=(8, 8))
ax = fig.gca(projection="3d")
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)


c = 1
y_equal = c - xs

ax.plot(xs, y_equal)
ax.elev = 45
ax.azim = -30

