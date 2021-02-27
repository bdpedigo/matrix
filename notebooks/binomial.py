#%%
import numpy as np
import matplotlib.pyplot as plt

n = 20
x = np.arange(0, n + 1)


def f(x, n):
    return (x / n) ** x * (1 - x / n) ** (n - x)


f(x, n)

plt.scatter(x, np.log(f(x, n)))
