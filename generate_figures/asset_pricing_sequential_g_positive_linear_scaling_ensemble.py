import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (
    zoomed_inset_axes,
    mark_inset,
    inset_axes,
)
import wandb
from utilities import get_results_by_tag, plot_params

output_dir = "./figures"
plot_name = "asset_pricing_sequential_g_positive_linear_scaling_ensemble"
output_path = output_dir + "/" + plot_name + ".pdf"
api = wandb.Api()
project = "highdimensionaleconlab/deep_learning_transversality"
params = plot_params((10, 4.5))


tag = "asset_pricing_sequential_g_positive_linear_scaling_ensemble"
quantiles = {"low_quant": 0.1, "mid_quant": 0.5, "high_quant": 0.9}

# getting artifacts and test_T/train_t_max from wandb

results = get_results_by_tag(api, project, tag, get_test_results=True)
assert results.id.nunique() == 100
parameters = get_results_by_tag(api, project, tag, get_config=True, max_runs=1)
max_T_test = int(parameters["test_T"])
max_T = int(parameters["train_t_max"])
t = np.array(range(0, max_T_test))

quant_result_g0 = (
    results.groupby("t")["p_rel_error", "p_t"]
    .quantile(list(quantiles.values()))
    .unstack(level=-1)
    .reset_index()
)
quant_result_g0["p_f_t"] = results.groupby("t")["p_f_t"].mean()

# plotting

plt.rcParams.update(params)

ax_p = plt.subplot(121)
plt.plot(quant_result_g0["p_t"][quantiles["mid_quant"]], "black", label=r"$\hat{p}(t)$")
plt.plot(quant_result_g0["p_f_t"], "black", label=r"$p_f(t)$", linestyle="dashed")
plt.fill_between(
    t,
    quant_result_g0["p_t"][quantiles["low_quant"]],
    quant_result_g0["p_t"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.15,
)
plt.axvline(x=max_T, color="0.0", linestyle="dashed")
ylim_min = 0.7 * np.amin(
    np.minimum(quant_result_g0["p_t"][quantiles["mid_quant"]], quant_result_g0["p_f_t"])
)
ylim_max = 1.1 * np.amax(
    np.maximum(quant_result_g0["p_t"][quantiles["mid_quant"]], quant_result_g0["p_f_t"])
)
plt.ylim([ylim_min, ylim_max])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), prop={"size": params["font.size"]})
plt.title(r"Prices: $\hat{p}(t)$")
plt.xlabel(r"Time($t$)")


ax_errors_p = plt.subplot(122, sharex=ax_p)
plt.plot(t, quant_result_g0["p_rel_error"][quantiles["mid_quant"]], "black")
plt.fill_between(
    t,
    quant_result_g0["p_rel_error"][quantiles["low_quant"]],
    quant_result_g0["p_rel_error"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.4,
)
plt.axvline(x=max_T, color="0.0", linestyle="dashed")
plt.title(r"Relative error: $\varepsilon_p(t)$")
plt.xlabel(r"Time($t$)")
plt.tight_layout()


# Zoom in for top left
time_window = [9, 12]
ave_value = 0.5 * (
    quant_result_g0["p_f_t"][time_window[0]] + quant_result_g0["p_f_t"][time_window[1]]
)
window_width = 0.03 * ave_value
matplotlib.rcParams.update({"ytick.labelsize": 8})

axins = zoomed_inset_axes(ax_p, 4, loc="lower right")
axins.plot(quant_result_g0["p_t"][quantiles["mid_quant"]], "black")
axins.plot(quant_result_g0["p_f_t"], "black", linestyle="dashed")
plt.fill_between(
    t,
    quant_result_g0["p_t"][quantiles["low_quant"]],
    quant_result_g0["p_t"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.5,
)


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
mark_inset(ax_p, axins, loc1=1, loc2=3, linewidth="0.7", ls="--", ec="0.5")


plt.savefig(output_path)
