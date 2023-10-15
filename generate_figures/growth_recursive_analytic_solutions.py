import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (
    zoomed_inset_axes,
    mark_inset,
    inset_axes,
)
import wandb

import sys
import os.path
import os
from utilities import tvc_violating_solution, plot_params

parent_dir = os.path.abspath(
    os.path.join(os.getcwd(), ".")
)  # going one level up in the directory
sys.path.insert(0, parent_dir)  # adding to the the path list

from growth_vfi import solve_growth_model_vfi


output_dir = "./figures"
plot_name = "growth_recursive_analytic_solutions"
output_path = output_dir + "/" + plot_name + ".pdf"
api = wandb.Api()
project = "highdimensionaleconlab/deep_learning_transversality"
params = plot_params((5, 4.5))  # I took these values from the previous version]

# define paramters to solve the model by VFI and to get a solution that violates TVC
alpha = 0.333333333333333
beta = 0.9
delta = 0.1
g = 0.0
maxk = 10
mink = 0.8

# paramters for a solution that violates TVC
T = 12
k0 = 0.8
c0 = 0.3


k_grid = np.linspace(
    mink,
    maxk,
    T,
)


# VFI solution
def f_scaled(x):
    return x**alpha


k_prime_vfi, c_vfi_func = solve_growth_model_vfi(
    k_grid,
    f_scaled,  # uses scaled f(k)
    beta,
    delta,
    g,
    c_solver_tol=1e-7,
)

k_vfi = np.empty_like(k_grid)
c_vfi = np.empty_like(k_grid)

for i in range(T):
    k_vfi[i] = k_prime_vfi(1.0, k_grid[i])
    c_vfi[i] = c_vfi_func(1.0, k_grid[i])


# Solution that violates transversality condition
k_tvc, c_tvc = tvc_violating_solution(T, alpha, beta, delta, k0, c0)


# start ploting
plt.rcParams.update(params)

plt.plot(k_tvc[0:11], k_tvc[1:12], "blue", label=r"$\tilde{k}'(k)$: Violating TVC")
plt.plot(k_grid, k_vfi, "black", label=r"$k'(k)$")
plt.plot(k_grid, k_grid, "black", linestyle="dashed", label="45 degree line")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), prop={"size": params["font.size"]})
plt.xlabel(r"Capital($k$)")
plt.legend(loc="upper left")
plt.tight_layout()


plt.savefig(output_path)
