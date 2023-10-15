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
plot_name = "growth_sequential_g0_grids"
output_path = output_dir + "/" + plot_name + ".pdf"
api = wandb.Api()
project = "highdimensionaleconlab/deep_learning_transversality"
params = plot_params((10, 9))

# One run with original grid. Comes from Figure 1 artifact and config

tag = "growth_sequential_g0_one_run"
results = get_results_by_tag(api, project, tag, get_test_results=True)
parameters = get_results_by_tag(api, project, tag, get_config=True)
assert results.id.nunique() == 1
max_T_test = int(parameters["test_T"])
max_T = int(parameters["train_t_max"]) + 1

k_sol = results["k_t_sol"]
c_sol = results["c_t_sol"]
k_0 = results["k_t_approx"]
c_0 = results["c_t_approx"]
k_0_error = results["k_rel_error"]
c_0_error = results["c_rel_error"]

t = np.array(range(0, max_T_test))

# One run with grid 1 artifact
tag = "growth_sequential_g0_grid_1_one_run"
results = get_results_by_tag(api, project, tag, get_test_results=True)
assert results.id.nunique() == 1

k_1 = results["k_t_approx"]
c_1 = results["c_t_approx"]
k_1_error = results["k_rel_error"]
c_1_error = results["c_rel_error"]

# One run with grid 2 artifact
tag = "growth_sequential_g0_grid_2_one_run"
results = get_results_by_tag(api, project, tag, get_test_results=True)
assert results.id.nunique() == 1
k_2 = results["k_t_approx"]
c_2 = results["c_t_approx"]
k_2_error = results["k_rel_error"]
c_2_error = results["c_rel_error"]


# Multiple runs with original grid and grid 2

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


# Grid 2 artifact
tag = "growth_sequential_g0_grid_2_ensemble"
results = get_results_by_tag(api, project, tag, get_test_results=True)
assert results.id.nunique() == 100

quant_result_g0_grid_2 = (
    results.groupby("t")[["c_rel_error", "c_t_approx", "k_rel_error", "k_t_approx"]]
    .quantile(list(quantiles.values()))
    .unstack(level=-1)
    .reset_index()
)

plt.rcParams.update(params)

ax_capital = plt.subplot(221)
plt.plot(k_sol, "black", linestyle="dashed", label=r"$k(t)$")
plt.plot(k_0, "black", label=r"$\hat{k}(t)$: Contiguous")
plt.plot(k_1, "lightblue", label=r"$\hat{k}(t)$: $\mathcal{X}_{train}$(Grid 1)")
plt.plot(k_2, "blue", label=r"$\hat{k}(t)$: $\mathcal{X}_{train}$(Grid 2)")
plt.axvline(x=max_T - 1, color="0.0", linestyle="dashed")
ylim_min = 0.8 * np.amin(np.minimum(np.minimum(k_sol, k_0), np.minimum(k_1, k_2)))
ylim_max = 1.3 * np.amax(np.maximum(np.maximum(k_sol, k_0), np.maximum(k_1, k_2)))
plt.ylim([ylim_min, ylim_max])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), prop={"size": params["font.size"]})
plt.legend(loc="upper left")
plt.title(r"Capital: $\hat{k}(t)$")
plt.xlabel(r"Time($t$)")
ax_capital.legend(fancybox=False, framealpha=1.0)


ax_errors = plt.subplot(222, sharex=ax_capital)
plt.plot(t, k_0_error, "black", label=r"$\varepsilon_k(t)$: Contiguous")
plt.plot(
    t,
    k_1_error,
    "lightblue",
    label=r"$\varepsilon_k(t)$: $\mathcal{X}_{train}$(Grid 1)",
)
plt.plot(
    t, k_2_error, "blue", label=r"$\varepsilon_k(t)$: $\mathcal{X}_{train}$(Grid 2)"
)
plt.axvline(x=max_T - 1, color="0.0", linestyle="dashed")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), prop={"size": params["font.size"]})
plt.title(r"Relative error: $\varepsilon_k(t)$")
plt.xlabel(r"Time($t$)")


ax_capital2 = plt.subplot(223, sharex=ax_capital)
plt.plot(k_sol, "black", linestyle="dashed", label=r"$k(t)$")
plt.plot(
    quant_result_g0["k_t_approx"][quantiles["mid_quant"]],
    "black",
    label=r"$\hat{k}(t)$: Contiguous",
)
plt.fill_between(
    t,
    quant_result_g0["k_t_approx"][quantiles["low_quant"]],
    quant_result_g0["k_t_approx"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.15,
)
plt.plot(
    quant_result_g0_grid_2["k_t_approx"][quantiles["mid_quant"]],
    "blue",
    label=r"$\hat{k}(t)$: $\mathcal{X}_{train}$(Grid 2)",
)
plt.fill_between(
    t,
    quant_result_g0_grid_2["k_t_approx"][quantiles["low_quant"]],
    quant_result_g0_grid_2["k_t_approx"][quantiles["high_quant"]],
    facecolor="blue",
    alpha=0.15,
)
plt.axvline(x=max_T - 1, color="0.0", linestyle="dashed")
ylim_min = 0.8 * np.amin(
    np.minimum(
        np.minimum(k_sol, quant_result_g0["k_t_approx"][quantiles["mid_quant"]]),
        np.minimum(k_sol, quant_result_g0_grid_2["k_t_approx"][quantiles["mid_quant"]]),
    )
)
ylim_max = 1.3 * np.amax(
    np.maximum(
        np.maximum(k_sol, quant_result_g0["k_t_approx"][quantiles["mid_quant"]]),
        np.maximum(k_sol, quant_result_g0_grid_2["k_t_approx"][quantiles["mid_quant"]]),
    )
)
plt.ylim([ylim_min, ylim_max])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(
    by_label.values(),
    by_label.keys(),
    prop={"size": params["font.size"]},
    loc="lower right",
)
plt.title(r"Capital: $\hat{k}(t)$")
plt.xlabel(r"Time($t$)")
ax_capital2.legend(fancybox=False, framealpha=1.0, loc="lower right")


ax_errors2 = plt.subplot(224, sharex=ax_capital)
plt.plot(
    t,
    quant_result_g0["k_rel_error"][quantiles["mid_quant"]],
    "black",
    label=r"$\varepsilon_p(t)$: Contiguous",
)
plt.fill_between(
    t,
    quant_result_g0["k_rel_error"][quantiles["low_quant"]],
    quant_result_g0["k_rel_error"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.4,
)
plt.plot(
    t,
    quant_result_g0_grid_2["k_rel_error"][quantiles["mid_quant"]],
    "blue",
    label=r"$\varepsilon_p(t)$: $\mathcal{X}_{train}$(Grid 2)",
)
plt.fill_between(
    t,
    quant_result_g0_grid_2["k_rel_error"][quantiles["low_quant"]],
    quant_result_g0_grid_2["k_rel_error"][quantiles["high_quant"]],
    facecolor="blue",
    alpha=0.1,
)
plt.axvline(x=max_T - 1, color="0.0", linestyle="dashed")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), prop={"size": params["font.size"]})
plt.title(r"Relative errors: $\varepsilon_k(t)$")
plt.xlabel(r"Time($t$)")
ax_errors2.legend(fancybox=False, framealpha=1.0)
plt.tight_layout()

# Zoom in for top left
time_window = [44, 49]
ave_value = 0.5 * (k_sol[time_window[0]] + k_sol[time_window[1]])
window_width = 0.02 * ave_value
matplotlib.rcParams.update({"ytick.labelsize": 8})

axins = zoomed_inset_axes(
    ax_capital,
    4,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.7, -0.3, 0.3),
    bbox_transform=ax_capital.transAxes,
)
axins.plot(k_sol, "black", linestyle="dashed")
axins.plot(k_0, "black")
axins.plot(k_1, "lightblue")
axins.plot(k_2, "blue")


plt.axvline(x=max_T, color="0.0", linestyle="dashed")
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
mark_inset(ax_capital, axins, loc1=1, loc2=3, linewidth="0.7", ls="--", ec="0.5")


# Zoom in for bottom left

time_window = [44, 49]
ave_value = 0.5 * (k_sol[time_window[0]] + k_sol[time_window[1]])
window_width = 0.02 * ave_value
matplotlib.rcParams.update({"ytick.labelsize": 8})

axins = zoomed_inset_axes(
    ax_capital2,
    4,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.7, -0.3, 0.3),
    bbox_transform=ax_capital2.transAxes,
)
axins.plot(k_sol, "black", linestyle="dashed")
axins.plot(k_1, "black")
axins.plot(k_2, "blue")
plt.fill_between(
    t,
    quant_result_g0["k_t_approx"][quantiles["low_quant"]],
    quant_result_g0["k_t_approx"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.4,
)
plt.fill_between(
    t,
    quant_result_g0_grid_2["k_t_approx"][quantiles["low_quant"]],
    quant_result_g0_grid_2["k_t_approx"][quantiles["high_quant"]],
    facecolor="blue",
    alpha=0.15,
)
plt.axvline(x=max_T, color="0.0", linestyle="dashed")
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
mark_inset(ax_capital2, axins, loc1=1, loc2=3, linewidth="0.7", ls="--", ec="0.5")


plt.savefig(output_path)
