import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import wandb
from mpl_toolkits.axes_grid1.inset_locator import (
    zoomed_inset_axes,
    mark_inset,
    inset_axes,
)
from utilities import get_results_by_tag, plot_params

output_dir = "./figures"
plot_name = "growth_recursive_g0_ensemble"
output_path = output_dir + "/" + plot_name + ".pdf"
tag = "growth_recursive_g0_ensemble"
api = wandb.Api()
project = "highdimensionaleconlab/deep_learning_transversality"
params = plot_params((10, 9))

#artifact and config
results = get_results_by_tag(api, project, tag, get_test_results=True)
assert results.id.nunique() == 100
parameters = get_results_by_tag(api, project, tag, get_config=True, max_runs = 1)
max_T_test = int(parameters["max_T_test"])
k_min = float(parameters["k_grid_min"])
k_max = float(parameters["k_grid_max"])

t = np.array(range(0, max_T_test))

quantiles = {"low_quant": 0.1, "mid_quant": 0.5, "high_quant": 0.9}
quant_result_g0 = (
    results.groupby("t")[["c_rel_error", "c_t_approx", "k_rel_error", "k_t_approx"]]
    .quantile(list(quantiles.values()))
    .unstack(level=-1)
    .reset_index()
)
quant_result_g0["k_t_sol"] = results.groupby("t")["k_t_sol"].mean()
quant_result_g0["c_t_sol"] = results.groupby("t")["c_t_sol"].mean()
quant_extrapolation = quant_result_g0[quant_result_g0["k_t_approx"][quantiles['mid_quant']] <= k_min]
quant_interpolation = quant_result_g0[quant_result_g0["k_t_approx"][quantiles['mid_quant']] >= k_min]



plt.rcParams.update(params)

ax_k = plt.subplot(221)
plt.plot(quant_result_g0["k_t_approx"][quantiles['mid_quant']], "black", label=r"$\hat{k}(t)$: median")
plt.plot(quant_result_g0["k_t_sol"], "black", label=r"$k(t)$", linestyle="dashed")
plt.fill_between(
    t, quant_result_g0["k_t_approx"][quantiles['low_quant']], quant_result_g0["k_t_approx"][quantiles['high_quant']], facecolor="black", alpha=0.15
)  # , label="$90/10\% interval$")

plt.fill_between(t, k_min, k_max, facecolor="0.9", label=r"Conv($\mathcal{D}$)")
ylim_min = 0.7 * np.amin(np.minimum(k_min, quant_result_g0["k_t_sol"]))
ylim_max = 1.1 * np.amax(np.maximum(k_max, quant_result_g0["k_t_sol"]))
plt.ylim([ylim_min, ylim_max])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), prop={"size": params['font.size']})
plt.title(r"Capital: $\hat{k}(t)$")
plt.xlabel(r"Time($t$)")

ax_errors_k = plt.subplot(222, sharex=ax_k)
plt.plot(
    quant_extrapolation["t"], 
    quant_extrapolation["k_rel_error"][quantiles['mid_quant']],
    "black",
    label=r"$\varepsilon_k(t)$: Extrapolation, median",
    linestyle="dashed",
)
plt.plot(
    quant_interpolation['t'],
    quant_interpolation["k_rel_error"][quantiles['mid_quant']],
    "black",
    label=r"$\varepsilon_k(t)$: Interpolation, median",
)

plt.fill_between(
    t, quant_result_g0["k_rel_error"][quantiles['low_quant']], quant_result_g0["k_rel_error"][quantiles['high_quant']], facecolor="black", alpha=0.4
) 
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), prop={"size": params['font.size']})
plt.title(r"Relative error: $\varepsilon_k(t)$")
plt.xlabel(r"Time($t$)")

ax_c = plt.subplot(223, sharex=ax_k)
plt.plot(quant_result_g0["c_t_approx"][quantiles['mid_quant']], "blue", label=r"$\hat{c}(t)$: median")
plt.plot(quant_result_g0["c_t_sol"], "blue", label=r"$c(t)$", linestyle="dashed")
plt.fill_between(
    t, quant_result_g0["c_t_approx"][quantiles['low_quant']], 
    quant_result_g0["c_t_approx"][quantiles['high_quant']], facecolor="blue", alpha=0.15
)
ylim_min = 0.7 * np.amin(np.minimum(quant_result_g0["c_t_approx"][quantiles['mid_quant']], quant_result_g0["c_t_sol"]))
ylim_max = 1.1 * np.amax(np.maximum(quant_result_g0["c_t_approx"][quantiles['mid_quant']], quant_result_g0["c_t_sol"]))
plt.ylim([ylim_min, ylim_max])
handlesc, labelsc = plt.gca().get_legend_handles_labels()
by_labelc = dict(zip(labelsc, handlesc))
plt.legend(by_labelc.values(), by_labelc.keys(), prop={"size":params['font.size']})
plt.title(r"Consumption: $\hat{c}(t)$")
plt.xlabel(r"Time($t$)")

ax_errors_k = plt.subplot(224, sharex=ax_k)
plt.plot(
    quant_extrapolation["t"], 
    quant_extrapolation["c_rel_error"][quantiles['mid_quant']],
    "blue",
    label=r"$\varepsilon_c(t)$: Extrapolation, median",
    linestyle="dashed",
)
plt.plot(
    quant_interpolation['t'],
    quant_interpolation["c_rel_error"][quantiles['mid_quant']],
    "blue",
    label=r"$\varepsilon_c(t)$: Interpolation, median",
)

plt.fill_between(
    t, quant_result_g0["c_rel_error"][quantiles['low_quant']], quant_result_g0["c_rel_error"][quantiles['high_quant']], facecolor="blue", alpha=0.4
) 
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), prop={"size": params['font.size']})
plt.title(r"Relative error: $\varepsilon_c(t)$")
plt.xlabel(r"Time($t$)")
plt.legend(loc="best")
plt.tight_layout()

# Zoom in for top left
time_window = [8, 12]
ave_value = 0.5 * (quant_result_g0["k_t_sol"][time_window[0]] + quant_result_g0["k_t_sol"][time_window[1]])
window_width = 0.0275 * ave_value
matplotlib.rcParams.update({"ytick.labelsize": 8})

axins = zoomed_inset_axes(ax_k, 4, loc="lower right")
axins.plot(quant_result_g0["k_t_approx"][quantiles['mid_quant']], "black")
axins.plot(quant_result_g0["k_t_sol"], "black", linestyle="dashed")
plt.fill_between(
    t, quant_result_g0["k_t_approx"][quantiles['low_quant']], quant_result_g0["k_t_approx"][quantiles['high_quant']], facecolor="black", alpha=0.5
)  # , label="$90/10\% interval$")

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
mark_inset(ax_k, axins, loc1=1, loc2=3, linewidth="0.7", ls="--", ec="0.5")


# Zoom in for bottom left
time_window = [8, 12]
ave_value = 0.5 * (quant_result_g0["c_t_sol"][time_window[0]] + quant_result_g0["c_t_sol"][time_window[1]])
window_width = 0.0275 * ave_value
matplotlib.rcParams.update({"ytick.labelsize": 8})

axins = zoomed_inset_axes(ax_c, 4, loc="center right")
axins.plot(quant_result_g0["c_t_approx"][quantiles['mid_quant']], "blue")
axins.plot(quant_result_g0["c_t_sol"], "blue", linestyle="dashed")
plt.fill_between(t, quant_result_g0["c_t_approx"][quantiles['low_quant']], quant_result_g0["c_t_approx"][quantiles['high_quant']], facecolor="blue", alpha=0.3)

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
mark_inset(ax_c, axins, loc1=1, loc2=3, linewidth="0.7", ls="--", ec="0.5")


plt.savefig(output_path)
