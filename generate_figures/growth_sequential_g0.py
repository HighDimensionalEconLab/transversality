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
plot_name = "growth_sequential_g0"
output_path = output_dir + "/" + plot_name + ".pdf"
api = wandb.Api()
project = "highdimensionaleconlab/deep_learning_transversality"
params = plot_params((10, 4.5))

# One run with original grid. Comes from Figure 1 artifact and config

tag = "growth_sequential_g0_one_run"
results = get_results_by_tag(api, project, tag, get_test_results=True)
parameters = get_results_by_tag(api, project, tag, get_config=True)
assert results.id.nunique() == 1
max_T_test = parameters["test_T"].item()
max_T = parameters["train_t_max"].item() 

k_sol = results["k_t_sol"]
c_sol = results["c_t_sol"]
t = np.array(range(0, max_T_test))
quantiles = {"low_quant": 0.1, "mid_quant": 0.5, "high_quant": 0.9}

# Original grid artifact
tag = "growth_sequential_g0_ensemble"
results = get_results_by_tag(api, project, tag, get_test_results=True)
assert results.id.nunique() == 100
quant_result_g0 = (
    results.groupby("t")[["c_rel_error", "c_t_approx", "k_rel_error", "k_t_approx"]]
    .quantile(list(quantiles.values()))
    .unstack(level=-1)
    .reset_index()
)



plt.rcParams.update(params)

ax_capital_consumption = plt.subplot(121)
plt.plot(quant_result_g0['k_t_approx'][quantiles["mid_quant"]], "black", label=r"$\hat{k}(t)$: median")
plt.plot(quant_result_g0['c_t_approx'][quantiles["mid_quant"]], "blue", label=r"$\hat{c}(t)$: median")
plt.plot(k_sol, "black", linestyle="dashed", label=r"$k(t)$")
plt.plot(c_sol, "black", linestyle="dashed", label=r"$c(t)$")
plt.fill_between(t,quant_result_g0["k_t_approx"][quantiles["low_quant"]], quant_result_g0["k_t_approx"][quantiles["high_quant"]], facecolor="black", alpha=0.15)
plt.fill_between(t,quant_result_g0["c_t_approx"][quantiles["low_quant"]], quant_result_g0["c_t_approx"][quantiles["high_quant"]], facecolor="blue", alpha=0.15)
plt.axvline(x=max_T, color="0.0", linestyle="dashed")
plt.legend(loc="upper left")
plt.title(r"Capital and Consumption: $\hat{k}(t)$ and $\hat{c}(t)$")
plt.xlabel(r"Time($t$)")
ylim_min = 0.5 * np.amin(c_sol)
ylim_max = 1.2 * np.amax(k_sol)
plt.ylim([ylim_min, ylim_max])
ax_capital_consumption.legend(fancybox=False, framealpha=1.0)


ax_rel_errors = plt.subplot(122)
plt.plot(quant_result_g0['k_rel_error'][quantiles["mid_quant"]], "black",label=r"$\varepsilon_k(t)$: median")
plt.fill_between(t,quant_result_g0["k_rel_error"][quantiles["low_quant"]], quant_result_g0["k_rel_error"][quantiles["high_quant"]], facecolor="black", alpha=0.15)
plt.plot(quant_result_g0['c_rel_error'][quantiles["mid_quant"]], "blue",label=r"$\varepsilon_c(t)$: median")
plt.fill_between(t,quant_result_g0["c_rel_error"][quantiles["low_quant"]], quant_result_g0["c_rel_error"][quantiles["high_quant"]], facecolor="blue", alpha=0.15)
plt.axvline(x=max_T, color="0.0", linestyle="dashed")
plt.title(r"Relative errors: $\varepsilon_k(t)$ and $\varepsilon_c(t)$")
plt.xlabel(r"Time($t$)")
plt.legend(loc="upper left")

plt.tight_layout()


# Zoom in for top left
time_window = [44, 49]
ave_value = 0.5 * (k_sol[time_window[0]] + k_sol[time_window[1]])
window_width = 0.02 * ave_value
matplotlib.rcParams.update({"ytick.labelsize": 8})

axins = zoomed_inset_axes(
    ax_capital_consumption,
    4,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.7, -0.3, 0.3),
    bbox_transform=ax_capital_consumption.transAxes,
)
axins.plot(k_sol, "black", linestyle="dashed")
plt.plot(quant_result_g0['k_t_approx'][quantiles["mid_quant"]], "black")
plt.fill_between(t,quant_result_g0["k_t_approx"][quantiles["low_quant"]], quant_result_g0["k_t_approx"][quantiles["high_quant"]], facecolor="black", alpha=0.35)
x1, x2, y1, y2 = (
    time_window[0],
    time_window[1],
    ave_value - window_width,
    ave_value + window_width,
)
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.xaxis.tick_top()
plt.xticks(fontsize=8, visible=False)
plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
plt.yticks(fontsize=8)
mark_inset(ax_capital_consumption, axins, loc1=1, loc2=3, linewidth="0.7", ls="--", ec="0.5")




plt.savefig(output_path)
