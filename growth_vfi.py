# Solves the optimal growth model. Code from Quantecon: https://python.quantecon.org/optgrowth.html
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from dataclasses import dataclass


# Utility to hold parameters for VFI algorithm
@dataclass
class VFIParameters:
    tol: float
    max_iter: int
    k_grid_size: int
    c_solver_tol: float
    k_min_multiplier: float
    k_max_multiplier: float
    interpolation_kind: str


# simple, slow VFI algorithm
def solve_growth_model_vfi(
    k_grid,
    f,
    beta,
    delta,
    g,
    vfi_tol=1e-9,
    max_iter=1000,
    c_solver_tol=1e-9,
    min_c=1e-9,
    interpolation_kind="cubic",
):
    min_k = k_grid[0]

    # The Bellman operator as a closure
    def T_growth(v):
        v_new = np.empty_like(v)
        k_prime = np.empty_like(v)
        c = np.empty_like(v)

        # value function used within optimization step
        v_interp = interp1d(
            k_grid, v, fill_value="extrapolate", kind=interpolation_kind
        )

        for j in np.arange(0, len(k_grid)):
            kt = k_grid[j]
            max_c = f(kt) + (1 - delta) * kt - min_k
            # Maximize RHS of Bellman equation

            def v_objective_min(c):  # reorganized to be a minimization problem
                ktp1 = (f(kt) + (1 - delta) * kt - c) / (1 + g)  # use LOM

                val = np.log(c) + beta * v_interp(ktp1)

                return -val  # converted to minimization problem

            result = minimize_scalar(
                v_objective_min,
                bounds=(min_c, max_c),
                method="bounded",
                options={"xatol": c_solver_tol},
            )
            c[j], v_max = result.x, -result.fun
            v_new[j] = v_max
            k_prime[j] = (f(kt) + (1 - delta) * kt - c[j]) / (1 + g)

        return k_prime, c, v_new

    v = np.zeros(len(k_grid))

    # Fixed point iteration on the bellman operator
    i = 0
    error = vfi_tol + 1
    while i < max_iter and error > vfi_tol:
        k_prime, c, v_new = T_growth(v)  # i.e. find v \approx T(v)

        error = np.mean(np.power(v - v_new, 2))
        i += 1
        v = v_new

    if i == max_iter:
        raise RuntimeError(f"Convergence failed after {i} iterations")

    # return back closures which rescale by the "z"
    k_prime_interp_np = interp1d(
        k_grid, k_prime, fill_value="extrapolate", kind=interpolation_kind
    )
    k_prime_interp = (
        lambda z, k: (1 + g) * z * k_prime_interp_np(k / z)
    )  # homogeneity of degree 1, k'(k,z) = z' * \hat{k}'(k/z)

    c_interp_np = interp1d(k_grid, c, fill_value="extrapolate", kind=interpolation_kind)
    c_interp = lambda z, k: z * c_interp_np(
        k / z
    )  #  homogeneity of degree 1, c(k,z) = z * \hat{c}(k/z)

    return k_prime_interp, c_interp
