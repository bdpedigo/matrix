#%%
import graspologic as gs 
import numpy as np 

B = np.array([[0.5, 0.1], [0.1, 0.4]])
n_per_comm = [100, 50]
A, labels = gs.simulations.sbm(n_per_comm, B, return_labels=True)

n_trials = 1000
p_hat = 0
B_hat = np.zeros_like(B)
for i in range(n_trials):
    np.random.shuffle(labels)
    model = gs.models.SBMEstimator(loops=False, directed=False)
    model.fit(A, labels)
    B_hat += model.block_p_

    er_model = gs.models.EREstimator(loops=False, directed=False)
    er_model.fit(A)
    p_hat += er_model.p_

p_hat = p_hat / n_trials
B_hat = B_hat / n_trials

print(p_hat)
print(B_hat)