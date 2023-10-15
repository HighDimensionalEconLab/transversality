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
plot_name = "growth_recursive_g0_multiple_steady_states_capital_consumption_ensemble"
output_path = output_dir + "/" + plot_name + ".pdf"
tag = "growth_recursive_multiple_ss"
api = wandb.Api()
project = "highdimensionaleconlab/deep_learning_transversality"
params = plot_params((10, 4.5))


# artifact and config
results = get_results_by_tag(api, project, tag, get_test_results=True) 
assert results.id.nunique() == 400  # number of runs 
results = results[results["retcode"] == 0]
parameters = get_results_by_tag(api, project, tag, get_config=True, max_runs=1)
max_T_test = int(parameters["max_T_test"])
k_min = float(parameters["k_grid_min"])
k_max = float(parameters["k_grid_max"])



t_0 = results[results["t"] == 0]
k_1_0 = t_0[t_0["k_t_approx"] == 0.4000000059604645]["id"].unique()
k_2_0 = t_0[t_0["k_t_approx"] == 1.0]["id"].unique()
k_3_0 = t_0[t_0["k_t_approx"] == 3.299999952316284]["id"].unique()
k_4_0 = t_0[t_0["k_t_approx"] == 4.0]["id"].unique()

k_1_runs = results[results["id"].isin(k_1_0)]
k_2_runs = results[results["id"].isin(k_2_0)]
k_3_runs = results[results["id"].isin(k_3_0)]
k_4_runs = results[results["id"].isin(k_4_0)]


high_steady_state_capital_value = results["k_ss_high_norm"][1]
low_steady_state_capital_value = results["k_ss_low_norm"][1]
high_steady_state_consumption_value = results["c_ss_high_norm"][1]
low_steady_state_consumption_value = results["c_ss_low_norm"][1]

t = np.array(range(0, max_T_test))

quantiles = {"low_quant": 0.1, "mid_quant": 0.5, "high_quant": 0.9}
results_quant = (
    results.groupby("t")[["c_rel_error", "k_rel_error"]]
    .quantile(list(quantiles.values()))
    .unstack(level=-1)
    .reset_index()
)

k_1_quant_result_g0 = (
    k_1_runs.groupby("t")[["c_rel_error", "c_t_approx", "k_rel_error", "k_t_approx"]]
    .quantile(list(quantiles.values()))
    .unstack(level=-1)
    .reset_index()
)
k_1_quant_result_g0["k_t_sol"] = k_1_runs.groupby("t")["k_t_sol"].median()
k_1_quant_result_g0["c_t_sol"] = k_1_runs.groupby("t")["c_t_sol"].median()

k_2_quant_result_g0 = (
    k_2_runs.groupby("t")[["c_rel_error", "c_t_approx", "k_rel_error", "k_t_approx"]]
    .quantile(list(quantiles.values()))
    .unstack(level=-1)
    .reset_index()
)
k_2_quant_result_g0["k_t_sol"] = k_2_runs.groupby("t")["k_t_sol"].median()
k_2_quant_result_g0["c_t_sol"] = k_2_runs.groupby("t")["c_t_sol"].median()


k_3_quant_result_g0 = (
    k_3_runs.groupby("t")[["c_rel_error", "c_t_approx", "k_rel_error", "k_t_approx"]]
    .quantile(list(quantiles.values()))
    .unstack(level=-1)
    .reset_index()
)
k_3_quant_result_g0["k_t_sol"] = k_3_runs.groupby("t")["k_t_sol"].median()
k_3_quant_result_g0["c_t_sol"] = k_3_runs.groupby("t")["c_t_sol"].median()

k_4_quant_result_g0 = (
    k_4_runs.groupby("t")[["c_rel_error", "c_t_approx", "k_rel_error", "k_t_approx"]]
    .quantile(list(quantiles.values()))
    .unstack(level=-1)
    .reset_index()
)
k_4_quant_result_g0["k_t_sol"] = k_3_runs.groupby("t")["k_t_sol"].median()
k_4_quant_result_g0["c_t_sol"] = k_3_runs.groupby("t")["c_t_sol"].median()



#Plotting starts

plt.rcParams.update(params)


ax_k = plt.subplot(121)
plt.plot(
    k_1_quant_result_g0["k_t_approx"][quantiles["mid_quant"]],
    "black",
    label=r"$\hat{k}(t)$",
)
plt.plot(k_1_quant_result_g0["k_t_sol"], "black", label=r"$k(t)$", linestyle="dashed")
plt.fill_between(
    t,
    k_1_quant_result_g0["k_t_approx"][quantiles["low_quant"]],
    k_1_quant_result_g0["k_t_approx"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.15,
)  # , label="$90/10\% interval$")

plt.plot(
    k_2_quant_result_g0["k_t_approx"][quantiles["mid_quant"]],
    "black",
    label=r"$\hat{k}(t)$",
)
plt.plot(k_2_quant_result_g0["k_t_sol"], "black", label=r"$k(t)$", linestyle="dashed")
plt.fill_between(
    t,
    k_2_quant_result_g0["k_t_approx"][quantiles["low_quant"]],
    k_2_quant_result_g0["k_t_approx"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.15,
)  # , label="$90/10\% interval$")

plt.axhline(y = low_steady_state_capital_value, color='red', linestyle="dashed", label="$k_1^*$")



plt.plot(
    k_3_quant_result_g0["k_t_approx"][quantiles["mid_quant"]],
    "black",
    label=r"$\hat{k}(t)$",
)
plt.plot(k_3_quant_result_g0["k_t_sol"], "black", label=r"$k(t)$", linestyle="dashed")
plt.fill_between(
    t,
    k_3_quant_result_g0["k_t_approx"][quantiles["low_quant"]],
    k_3_quant_result_g0["k_t_approx"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.15,
)  # , label="$90/10\% interval$")

plt.plot(
    k_4_quant_result_g0["k_t_approx"][quantiles["mid_quant"]],
    "black",
    label=r"$\hat{k}(t)$",
)
plt.plot(k_4_quant_result_g0["k_t_sol"], "black", label=r"$k(t)$", linestyle="dashed")
plt.fill_between(
    t,
    k_4_quant_result_g0["k_t_approx"][quantiles["low_quant"]],
    k_4_quant_result_g0["k_t_approx"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.15,
)  # , label="$90/10\% interval$")

plt.axhline(y = high_steady_state_capital_value, color='red', linestyle="dotted", label="$k_2^*$")


ylim_min = 0.7 * np.amin(results["k_t_sol"])
ylim_max = 1.1 * np.amax(results["k_t_sol"])
plt.ylim([ylim_min, ylim_max])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), prop={"size": params["font.size"]})
plt.title(r"Capital: $\hat{k}(t)$")
plt.xlabel(r"Time($t$)")


ax_c = plt.subplot(122, sharex=ax_k)
plt.plot(
    k_1_quant_result_g0["c_t_approx"][quantiles["mid_quant"]],
    "blue",
    label=r"$\hat{c}(t)$",
)
plt.plot(k_1_quant_result_g0["c_t_sol"], "blue", label=r"$c(t)$", linestyle="dashed")
plt.fill_between(
    t,
    k_1_quant_result_g0["c_t_approx"][quantiles["low_quant"]],
    k_1_quant_result_g0["c_t_approx"][quantiles["high_quant"]],
    facecolor="blue",
    alpha=0.15,
)  # , label="$90/10\% interval$")

plt.plot(
    k_2_quant_result_g0["c_t_approx"][quantiles["mid_quant"]],
    "blue",
    label=r"$\hat{c}(t)$",
)
plt.plot(k_2_quant_result_g0["c_t_sol"], "blue", label=r"$c(t)$", linestyle="dashed")
plt.fill_between(
    t,
    k_2_quant_result_g0["c_t_approx"][quantiles["low_quant"]],
    k_2_quant_result_g0["c_t_approx"][quantiles["high_quant"]],
    facecolor="blue",
    alpha=0.15,
)  # , label="$90/10\% interval$")

plt.axhline(y = low_steady_state_consumption_value, color='red', linestyle="dashed", label="$c_1^*$")


plt.plot(
    k_3_quant_result_g0["c_t_approx"][quantiles["mid_quant"]],
    "blue",
    label=r"$\hat{c}(t)$",
)
plt.plot(k_3_quant_result_g0["c_t_sol"], "blue", label=r"$c(t)$", linestyle="dashed")
plt.fill_between(
    t,
    k_3_quant_result_g0["c_t_approx"][quantiles["low_quant"]],
    k_3_quant_result_g0["c_t_approx"][quantiles["high_quant"]],
    facecolor="blue",
    alpha=0.15,
)  # , label="$90/10\% interval$")

plt.plot(
    k_4_quant_result_g0["c_t_approx"][quantiles["mid_quant"]],
    "blue",
    label=r"$\hat{c}(t)$",
)
plt.plot(k_4_quant_result_g0["c_t_sol"], "blue", label=r"$c(t)$", linestyle="dashed")
plt.fill_between(
    t,
    k_4_quant_result_g0["c_t_approx"][quantiles["low_quant"]],
    k_4_quant_result_g0["c_t_approx"][quantiles["high_quant"]],
    facecolor="blue",
    alpha=0.15,
)  # , label="$90/10\% interval$")

plt.axhline(y = high_steady_state_consumption_value, color='red', linestyle="dotted", label="$c_2^*$")

ylim_min = 0.7 * np.amin(
    np.amin(
        results["c_t_sol"],
    )
)
ylim_max = 1.1 * np.amax(
    np.amax(
        results["c_t_sol"],
    )
)
plt.ylim([ylim_min, ylim_max])
handlesc, labelsc = plt.gca().get_legend_handles_labels()
by_labelc = dict(zip(labelsc, handlesc))
plt.legend(by_labelc.values(), by_labelc.keys(), prop={"size": params["font.size"]})
plt.title(r"Consumption: $\hat{c}(t)$")
plt.xlabel(r"Time($t$)")


## Zoom in parts

    ## zoom in capital
k_t_3_med = k_3_quant_result_g0["k_t_approx"][quantiles["mid_quant"]]
time_window = [10, 15]
ave_value = 0.5 * (k_t_3_med[time_window[0]] + k_t_3_med[time_window[1]])
window_width = 0.035 * ave_value
matplotlib.rcParams.update({"ytick.labelsize": 8})

axins = zoomed_inset_axes(ax_k, 4, loc="center")
axins.plot(k_3_quant_result_g0["k_t_approx"][quantiles["mid_quant"]], "black")
axins.plot(k_3_quant_result_g0["k_t_sol"], "black",  linestyle="dashed")
axins.fill_between(
    t,
    k_3_quant_result_g0["k_t_approx"][quantiles["low_quant"]],
    k_3_quant_result_g0["k_t_approx"][quantiles["high_quant"]],
    facecolor="black",
    alpha=0.15,
)  

axins.axhline(y = high_steady_state_capital_value, color='red', linestyle="dotted")


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
mark_inset(ax_k, axins, loc1=1, loc2=3, linewidth="0.7", ls="--", ec="0.5")

    ## Zoom in for consumption

c_t_3_med = k_3_quant_result_g0["c_t_approx"][quantiles["mid_quant"]]
time_window = [10, 15]
ave_value = 0.5 * (c_t_3_med[time_window[0]] + c_t_3_med[time_window[1]])
window_width = 0.03 * ave_value
matplotlib.rcParams.update({"ytick.labelsize": 8})

axins = zoomed_inset_axes(ax_c, 4, loc="center")
axins.plot(k_3_quant_result_g0["c_t_approx"][quantiles["mid_quant"]], "blue")
axins.plot(k_3_quant_result_g0["c_t_sol"], "blue",  linestyle="dashed")
axins.fill_between(
    t,
    k_3_quant_result_g0["c_t_approx"][quantiles["low_quant"]],
    k_3_quant_result_g0["c_t_approx"][quantiles["high_quant"]],
    facecolor="blue",
    alpha=0.35,
)  
axins.axhline(y = high_steady_state_consumption_value, color='red', linestyle="dotted")


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