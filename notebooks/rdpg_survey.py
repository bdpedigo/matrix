#%% [markdown]
# Notes on RDPG Survey

#%% [markdown]


# %% [markdown]
# Let $U$ and $V$ be two $n \times d$ matrices with orthonormal columns. Then the vector
# of $d$ cannonical or principle angles between their column spaces is a vector $\Theta$
# with $$cos(\Theta) = [\sigma_1, ..., \sigma_d]^T$$ where $\sigma_1, ..., \sigma_d$ come
# from the SVD of $U^TV$. Thus $$\Theta = [cos^{-1}(\sigma_1), ..., cos^{-1}(\sigma_d)]$$
# Then the *matrix* $sin(\Theta)$ is the diagonal matrix where $sin(\Theta)_{ii}
# = sin(\theta_i)$.

#%% [markdown]
## One form of Davis-Kahan
# *preliminaries*  testing $$\|sin(\Theta)\|_F \leq \frac{2 \sqrt{d} \|V - V'\|}{\lambda_d(H) - \lambda_{d+1}(H)}$$


#%% [markdown]
## Consistency of latent position estimates (Thm. 26)

#%% [markdown]
## CLT for estimated latent positions
