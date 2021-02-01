#%%

import numpy as np
import pandas as pd

p = 0.75
n = 3
n_sims = 10000

rows = []
for p in np.linspace(0, 1, 100):
    majority_correct = 0
    simple_correct = 0
    for i in range(n_sims):
        n_correct = np.random.binomial(n, p)
        if n_correct > 1:
            majority_correct += 1

        simple_flip = np.random.binomial(1, p)
        if simple_flip > 0:
            simple_correct += 1
    majority_correct /= n_sims
    simple_correct /= n_sims
    rows.append({"p_correct": majority_correct, "method": "majority", "p": p})
    rows.append({"p_correct": simple_correct, "method": "simple", "p": p})
results = pd.DataFrame(rows)
import matplotlib.pyplot as plt
import seaborn as sns

sns.lineplot(data=results, x="p", y="p_correct", hue="method")
