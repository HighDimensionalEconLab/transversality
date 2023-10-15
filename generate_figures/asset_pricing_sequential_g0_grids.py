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
plot_name = "asset_pricing_sequential_g0_grids"
output_path = output_dir + "/" + plot_name + ".pdf"
api = wandb.Api()
project = "highdimensionaleconlab/deep_learning_transversality"
params = plot_params((10, 9))

# One run with original grid. Comes from Figure 1 artifact and config

tag = "asset_pricing_sequential_g0_one_run"
results = get_results_by_tag(api, project, tag, get_test_results=True)
parameters = get_results_by_tag(api, project, tag, get_config=True)
assert results.id.nunique() == 1
max_T_test = int(parameters["test_T"])
max_T = int(parameters["train_t_max"])

p_f = results["p_f_t"]
p_0 = results["p_t"]
p_0_error = results["p_rel_error"]

t = np.array(range(0, max_T_test))

# One run with grid 1 artifact
tag = "asset_pricing_sequential_g0_grid_1_one_run"
results = get_results_by_tag(api, project, tag, get_test_results=True)
assert results.id.nunique() == 1
p_1 = results["p_t"]
p_1_error = results["p_rel_error"]

# One run with grid 2 artifact
tag = "asset_pricing_sequential_g0_grid_2_one_run"
results = get_results_by_tag(api, project, tag, get_test_results=True)
assert results.id.nunique() == 1
p_2 = results["p_t"]
p_2_error = results["p_rel_error"]


# Multiple runs with original grid and grid 2

quantiles = {"low_quant": 0.1, "mid_quant": 0.5, "high_quant": 0.9}

# Original grid artifact
tag = "asset_pricing_sequential_g0_ensemble"
results = get_results_by_tag(api, project, tag, get_test_results=True)
assert results.id.nunique() == 100
quant_result_g0 = (
    results.groupby("t")["p_rel_error", "p_t"]
    .quantile(list(quantiles.values()))
    .unstack(level=-1)
    .reset_index()
)


# Grid 2 artifact
tag = "asset_pricing_sequential_g0_grid_2_ensemble"
results = get_results_by_tag(api, project, tag, get_test_results=True)
assert results.id.nunique() == 100

quant_result_g0_grid_2 = (
    results.groupby("t")["p_rel_error", "p_t"]
    .quantile(list(quantiles.values()))
    .unstack(level=-1)
    .reset_index()
)

# plotting

plt.rcParams.update(params)

ax_prices = plt.subplot(221)
plt.plot(p_f, "black", linestyle="dashed", label=r"$p_f(t)$")
plt.plot(p_0, "black", label=r"$\hat{p}(t)$: Contiguous")
plt.plot(p_1, "lightblue", label=r"$\hat{p}(t)$: $\mathcal{X}_{train}$(Grid 1)")
plt.plot(p_2, "blue", label=r"$\hat{p}(t)$: $\mathcal{X}_{train}$(Grid 2)")
plt.axvline(x=max_T, color="0.0", linestyle="dashed")
ylim_min = 0.9 * np.amin(np.minimum(np.minimum(p_f, p_0), np.minimum(p_1, p_2)))
ylim_max = 1.1 * np.amax(np.maximum(np.maximum(p_f, p_0), np.maximum(p_1, p_2)))
plt.ylim([ylim_min, ylim_max])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), prop={"size": params["font.size"]})
plt.title(r"Prices: $\hat{p}(t)$")
plt.xlabel(r"Time($t$)")
ax_prices.legend(fancybox=False, framealpha=1.0)
plt.legend(loc="lower left")

ax_errors = plt.subplot(222, sharex=ax_prices)
plt.plot(t, p_0_error, "black", label=r"$\varepsilon_p(t)$: Contiguous")
plt.plot(
    t,
    p_1_error,
    "lightblue",
    label=r"$\varepsilon_p(t)$: $\mathcal{X}_{train}$(Grid 1)",
)
plt.plot(
    t, p_2_error, "blue", label=r"$\varepsilon_p(t)$: $\mathcal{X}_{train}$(Grid 2)"
)
plt.axvline(x=max_T, color="0.0", linestyle="dashed")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.title(r"Relative errors: $\varepsilon_p(t)$")
plt.xlabel(r"Time($t$)")
ax_errors.legend(fancybox=False, framealpha=1.0)
quant_result_g0_grid_2["p_t"][quantiles["low_quant"]]

ax_prices2 = plt.subplot(223, sharex=ax_prices)
plt.plot(p_f, "black", linestyle="dashed", label=r"$p_f(t)$")
plt.plot(
    quant_result_g0["p_t"][quantiles["mid_quant"]],
    "black",
    label=r"$\hat{p}(t)$: Contiguous",
)
plt.fill_between(
    t,
    quant_result_g0["p_t"][quantiles["low_quant"]],
    quant_result_g0["p_t"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.15,
)

plt.plot(
    quant_result_g0_grid_2["p_t"][quantiles["mid_quant"]],
    "blue",
    label=r"$\hat{p}(t)$: $\mathcal{X}_{train}$(Grid 2)",
)
plt.fill_between(
    t,
    quant_result_g0_grid_2["p_t"][quantiles["low_quant"]],
    quant_result_g0_grid_2["p_t"][quantiles["high_quant"]],
    facecolor="blue",
    alpha=0.15,
)
plt.axvline(x=max_T, color="0.0", linestyle="dashed")
ylim_min = 0.9 * np.amin(
    np.minimum(
        np.minimum(p_f, quant_result_g0_grid_2["p_t"][quantiles["mid_quant"]]),
        np.minimum(p_f, quant_result_g0_grid_2["p_t"][quantiles["mid_quant"]]),
    )
)
ylim_max = 1.1 * np.amax(
    np.maximum(
        np.maximum(p_f, quant_result_g0_grid_2["p_t"][quantiles["mid_quant"]]),
        np.maximum(p_f, quant_result_g0_grid_2["p_t"][quantiles["mid_quant"]]),
    )
)
plt.ylim([ylim_min, ylim_max])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), prop={"size": params["font.size"]})
plt.title(r"Prices: $\hat{p}(t)$")
plt.xlabel(r"Time($t$)")
plt.legend(loc="lower left")


ax_errors2 = plt.subplot(224, sharex=ax_prices)
plt.plot(
    t,
    quant_result_g0["p_rel_error"][quantiles["mid_quant"]],
    "black",
    label=r"$\varepsilon_p(t)$: Contiguous",
)
plt.fill_between(
    t,
    quant_result_g0["p_rel_error"][quantiles["low_quant"]],
    quant_result_g0["p_rel_error"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.4,
)
plt.plot(
    t,
    quant_result_g0_grid_2["p_rel_error"][quantiles["mid_quant"]],
    "blue",
    label=r"$\varepsilon_p(t)$: $\mathcal{X}_{train}$(Grid 2)",
)
plt.fill_between(
    t,
    quant_result_g0_grid_2["p_rel_error"][quantiles["low_quant"]],
    quant_result_g0_grid_2["p_rel_error"][quantiles["high_quant"]],
    facecolor="blue",
    alpha=0.1,
)
plt.axvline(x=max_T, color="0.0", linestyle="dashed")
plt.title(r"Relative errors: $\varepsilon_p(t)$")
plt.xlabel(r"Time($t$)")
plt.legend(loc="best")
plt.tight_layout()

# Zoom in for top left
time_window = [24, 29]
ave_value = 0.5 * (p_f[time_window[0]] + p_f[time_window[1]])
window_width = 0.007 * ave_value
matplotlib.rcParams.update({"ytick.labelsize": 8})

axins = zoomed_inset_axes(
    ax_prices,
    4,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.7, -0.3, 0.3),
    bbox_transform=ax_prices.transAxes,
)
axins.plot(p_f, "black", linestyle="dashed")
axins.plot(p_0, "black")
axins.plot(p_1, "lightblue")
axins.plot(p_2, "blue")

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
mark_inset(ax_prices, axins, loc1=1, loc2=3, linewidth="0.7", ls="--", ec="0.5")


# Zoom in for bottom left

time_window = [24, 29]
ave_value = 0.5 * (p_f[time_window[0]] + p_f[time_window[1]])
window_width = 0.007 * ave_value
matplotlib.rcParams.update({"ytick.labelsize": 8})

axins = zoomed_inset_axes(
    ax_prices2,
    4,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.7, -0.3, 0.3),
    bbox_transform=ax_prices2.transAxes,
)
axins.plot(p_f, "black", linestyle="dashed")
axins.plot(quant_result_g0["p_t"][quantiles["mid_quant"]], "black")
axins.plot(quant_result_g0_grid_2["p_t"][quantiles["mid_quant"]], "blue")
plt.fill_between(
    t,
    quant_result_g0["p_t"][quantiles["low_quant"]],
    quant_result_g0["p_t"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.4,
)
plt.fill_between(
    t,
    quant_result_g0_grid_2["p_t"][quantiles["low_quant"]],
    quant_result_g0_grid_2["p_t"][quantiles["high_quant"]],
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
mark_inset(ax_prices2, axins, loc1=1, loc2=3, linewidth="0.7", ls="--", ec="0.5")

plt.savefig(output_path)
