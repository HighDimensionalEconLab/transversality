import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utilities import plot_params, tvc_violating_solution


import sys
import os.path
import os

parent_dir = os.path.abspath(
    os.path.join(os.getcwd(), ".")
)  # going one level up in the directory
sys.path.insert(0, parent_dir)  # adding to the the path list

from growth_vfi import solve_growth_model_vfi

output_dir = "./figures"
plot_name = "growth_sequential_violating_tvc_paths_below"
output_path = output_dir + "/" + plot_name + ".pdf"


T = 100
alpha = 0.333333333333333
beta = 0.9
delta = 0.1
g = 0.0

# From below
k0 = 0.8
c0 = 0.376


# Steady-state levels
k_star = ((((1 + g) / beta) - 1.0 + delta) / alpha) ** (1 / (alpha - 1))

c_star = k_star**alpha - delta * k_star

k_tilde_max = delta ** (1 / (alpha - 1))

# VFI solution

maxk = 1.1 * max(k0, k_star)
mink = 0.9 * min(k0, k_star)

# VFI solution

maxk = 1.1 * max(k0, k_star)
mink = 0.9 * min(k0, k_star)


def f_scaled(x):
    return x**alpha


k_grid = np.linspace(
    mink,
    maxk,
    T,
)

k_prime_vfi, c_vfi_func = solve_growth_model_vfi(
    k_grid,
    f_scaled,  # uses scaled f(k)
    beta,
    delta,
    g,
    c_solver_tol=1e-7,
)

kpath = np.zeros([int(T)])
cpath = np.zeros_like(kpath)

kpath[0] = k0
for t_val in range(T):
    if t_val < T - 1:
        kpath[t_val + 1] = k_prime_vfi(1.0, kpath[t_val])
    cpath[t_val] = c_vfi_func(1.0, kpath[t_val])

# Solution that violates transversality condition
k_tvc, c_tvc = tvc_violating_solution(
    T, alpha, beta, delta, k0, c0
)  # this function is defined in the utilities.py


params = plot_params((10, 4.5))
plt.rcParams.update(params)

# Plotting the violation of TVC with initial condition below the steady state
ax_k = plt.subplot(131)
plt.plot(k_tvc, "blue", label=r"$\tilde{k}(t)$")
plt.plot(kpath, "black", label=r"$k(t)$")
plt.plot(k_star * np.ones(T), "red", linestyle="dashed", label=r"$k^*$")
plt.plot(
    k_tilde_max * np.ones(T),
    "lightblue",
    linestyle="dashed",
    label=r"$\tilde{k}_{max}$",
)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.title("Capital")
plt.xlabel(r"Time($t$)")
plt.tight_layout()


ax_c = plt.subplot(132, sharex=ax_k)
plt.plot(c_tvc, "blue", label=r"$\tilde{c}(t)$")
plt.plot(cpath, "black", label=r"$c(t)$")
plt.plot(c_star * np.ones(T), "red", linestyle="dashed", label=r"$c^*$")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.title("Consumption")
plt.xlabel(r"Time($t$)")
plt.legend(loc="best")
plt.tight_layout()

ax_u_prime_c = plt.subplot(133)
plt.plot(1.0 / c_tvc, "blue", label=r"$u'(\tilde{c}(t))$")
plt.plot(1.0 / cpath, "black", label=r"$u'(c(t))$")
plt.plot(1.0 / c_star * np.ones(T), "red", linestyle="dashed", label=r"$u'(c^*)$")
ax_u_prime_c.set_yscale("log")
plt.title("Marginal utility of consumption")
plt.xlabel(r"Time($t$)")
plt.legend(loc="best")
plt.tight_layout()

plt.savefig(output_path)
