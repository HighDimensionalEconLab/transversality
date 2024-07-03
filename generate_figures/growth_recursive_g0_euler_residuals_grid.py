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
from utilities import get_results_by_tag, plot_params

output_dir = "./figures"
plot_name = "growth_recursive_g0_euler_residuals_grid"
output_path = output_dir + "/" + plot_name + ".pdf"
api = wandb.Api()
project = "highdimensionaleconlab/deep_learning_transversality"
params = plot_params((10, 9))


# Multiple runs with NN(k) and NN(c)

quantiles = {"low_quant": 0.1, "mid_quant": 0.5, "high_quant": 0.9}

# Original NN(k) artifact
tag = "growth_recursive_g0_grid_ensemble"
results = get_results_by_tag(api, project, tag, get_test_results=True)
assert results.id.nunique() == 100
results["res_t"] = np.power(results["res_t"], 2.0)
quant_result_g0_k = (
    results.groupby("k_t")[["res_t", "k_t", "k_tp1"]]
    .quantile(list(quantiles.values()))
    .unstack(level=-1)
    .reset_index()
)


# NN(c) artifact
tag = "growth_recursive_g0_using_c_grid_ensemble"
results = get_results_by_tag(api, project, tag, get_test_results=True)
assert results.id.nunique() == 100
results["res_t"] = np.power(results["res_t"], 2.0)
quant_result_g0_c = (
    results.groupby("k_t")[["res_t", "k_t", "k_tp1"]]
    .quantile(list(quantiles.values()))
    .unstack(level=-1)
    .reset_index()
)

plt.rcParams.update(params)

ax_errors_k = plt.subplot(221)

plt.plot(
    quant_result_g0_c["k_t"][quantiles["mid_quant"]],
    quant_result_g0_k["res_t"][quantiles["mid_quant"]],
    "black", 
    label = r"Euler residuals squared: median"
)

plt.fill_between(
    quant_result_g0_k["k_t"][quantiles["mid_quant"]],
    quant_result_g0_k["res_t"][quantiles["low_quant"]],
    quant_result_g0_k["res_t"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.15,
)
min_y = min(
    quant_result_g0_k["res_t"][quantiles["low_quant"]].min(),
    quant_result_g0_c["res_t"][quantiles["low_quant"]].min()
)
max_y = max(
    quant_result_g0_k["res_t"][quantiles["high_quant"]].max(),
    quant_result_g0_c["res_t"][quantiles["high_quant"]].max()
)

# Set the common y-axis limits
ax_errors_k.set_ylim(min_y, max_y)
ax_errors_k.set_yscale("log")

plt.title(r"Euler residuals squared: approximating $k'(k)$")
plt.xlabel(r"capital($k$)")
plt.legend()


ax_errors_c = plt.subplot(222)

plt.plot(
    quant_result_g0_c["k_t"][quantiles["mid_quant"]],
    quant_result_g0_c["res_t"][quantiles["mid_quant"]],
    "black",
    label= r"Euler residuals squared: median"
)

plt.fill_between(
    quant_result_g0_c["k_t"][quantiles["mid_quant"]],
    quant_result_g0_c["res_t"][quantiles["low_quant"]],
    quant_result_g0_c["res_t"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.15,
)

ax_errors_c.set_yscale("log")
ax_errors_c.set_ylim(min_y, max_y)
plt.title(r"Euler residuals squared: approximating $c(k)$")
plt.xlabel(r"capital($k$)")
plt.legend()


ax_capital_k = plt.subplot(223)
plt.plot(
    quant_result_g0_k["k_t"][quantiles["mid_quant"]],
    quant_result_g0_k["k_tp1"][quantiles["mid_quant"]],
    "black",
    label=r"$k'(k)$: median",
)
plt.plot(
    quant_result_g0_k["k_t"][quantiles["mid_quant"]],
    quant_result_g0_k["k_t"][quantiles["mid_quant"]],
    "black",
    linestyle="dashed",
    label="45 degree line",
)

plt.fill_between(
    quant_result_g0_k["k_t"][quantiles["mid_quant"]],
    quant_result_g0_k["k_tp1"][quantiles["low_quant"]],
    quant_result_g0_k["k_tp1"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.15,
)
ylim_min = 0.8 * np.amin(
    np.minimum(
        quant_result_g0_k["k_tp1"][quantiles["mid_quant"]],
        quant_result_g0_c["k_tp1"][quantiles["mid_quant"]],
    )
)
ylim_max = 1.3 * np.amax(
    np.maximum(
        quant_result_g0_k["k_tp1"][quantiles["mid_quant"]],
        quant_result_g0_c["k_tp1"][quantiles["mid_quant"]],
    )
)
plt.ylim([ylim_min, ylim_max])

plt.title(r"$k'(k)$: appriximating  $k'(k)$")
plt.xlabel(r"capital($k$)")
plt.legend()

ax_capital_c = plt.subplot(224)
plt.plot(
    quant_result_g0_c["k_t"][quantiles["mid_quant"]],
    quant_result_g0_c["k_tp1"][quantiles["mid_quant"]],
    "black",
    label=r"$k'(k)$: median",
)
plt.plot(
    quant_result_g0_c["k_t"][quantiles["mid_quant"]],
    quant_result_g0_c["k_t"][quantiles["mid_quant"]],
    "black",
    linestyle="dashed",
    label="45 degree line",
)

plt.fill_between(
    quant_result_g0_c["k_t"][quantiles["mid_quant"]],
    quant_result_g0_c["k_tp1"][quantiles["low_quant"]],
    quant_result_g0_c["k_tp1"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.15,
)
ylim_min = 0.8 * np.amin(
    np.minimum(
        quant_result_g0_k["k_tp1"][quantiles["mid_quant"]],
        quant_result_g0_c["k_tp1"][quantiles["mid_quant"]],
    )
)
ylim_max = 1.3 * np.amax(
    np.maximum(
        quant_result_g0_k["k_tp1"][quantiles["mid_quant"]],
        quant_result_g0_c["k_tp1"][quantiles["mid_quant"]],
    )
)
plt.ylim([ylim_min, ylim_max])


plt.title(r"$k'(k)$: approximating $c(k)$")
plt.xlabel(r"capital($k$)")
plt.legend()
plt.tight_layout()


plt.savefig(output_path)
