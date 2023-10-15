import csv
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

# Figure 25
output_dir = "./figures"
plot_name = "growth_sequential_multiple_steady_states_var_initial_k_0_full"
output_path = output_dir + "/" + plot_name + ".pdf"
api = wandb.Api()
project = "highdimensionaleconlab/deep_learning_transversality"
params = plot_params((10, 9))
tag = "growth_multiple_steady_states_var_initial_k_0"
results = get_results_by_tag(api, project, tag, get_test_results=True)

parameters = get_results_by_tag(api, project, tag, get_config=True)
max_T_test = int(parameters["test_T"][0])
max_T = int(parameters["train_t_max"][0]) + 1
t = np.array(range(0, max_T_test))

runs = results["id"].unique()

c_low = results["c_ss_low_norm"].iloc[0]
c_high = results["c_ss_high_norm"].iloc[0]


plt.rcParams.update(params)
ax_capital = plt.subplot(211)
for id in runs:
    res = results[results["id"] == id]["c_t_approx"]
    # grabbing the wrong trajectories
    if (
        res.iloc[0] > c_low
        and res.iloc[49] < (c_high - 0.1)
        and res.iloc[1] - res.iloc[0] > 0
    ):
        plt.plot(t, results[results["id"] == id]["k_t_approx"], "red")
    else:
        plt.plot(t, results[results["id"] == id]["k_t_approx"], "gray")

plt.plot(
    t,
    results[results["id"] == id]["k_ss_low_norm"],
    "black",
    linestyle="dashed",
    label="$k_1^*$",
)
plt.plot(
    t,
    results[results["id"] == id]["k_ss_high_norm"],
    "black",
    linestyle="dotted",
    label="$k_2^*$",
)

plt.axvline(x=max_T - 1, color="0.0", linestyle="dashed")
ylim_min = 0.9 * np.amin(results["k_t_approx"])
ylim_max = 1.1 * np.amax(results["k_t_approx"])
plt.ylim([ylim_min, ylim_max])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(
    by_label.values(),
    by_label.keys(),
    prop={"size": params["font.size"]},
    loc="upper right",
)
plt.title(r"Capital: $\hat{k}(t)$")
plt.xlabel(r"Time($t$)")
plt.tight_layout()

ax_consumption = plt.subplot(212, sharex=ax_capital)
for id in runs:
    res = results[results["id"] == id]["c_t_approx"]
    if (
        res.iloc[0] > c_low
        and res.iloc[49] < (c_high - 0.1)
        and res.iloc[1] - res.iloc[0] > 0
    ):
        plt.plot(t, results[results["id"] == id]["c_t_approx"], "red")
    else:
        plt.plot(t, results[results["id"] == id]["c_t_approx"], "blue", alpha=0.5)

plt.plot(
    t,
    results[results["id"] == id]["c_ss_low_norm"],
    "black",
    linestyle="dashed",
    label="$c_1^*$",
)
plt.plot(
    t,
    results[results["id"] == id]["c_ss_high_norm"],
    "black",
    linestyle="dotted",
    label="$c_2^*$",
)
plt.axvline(x=max_T - 1, color="0.0", linestyle="dashed")
ylim_min = 0.9 * np.amin(results["c_t_approx"])
ylim_max = 1.1 * np.amax(results["c_t_approx"])
plt.ylim([ylim_min, ylim_max])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(
    by_label.values(),
    by_label.keys(),
    prop={"size": params["font.size"]},
    loc="upper right",
)
plt.title(r"Consumption: $\hat{c}(t)$")
plt.xlabel(r"Time($t$)")
plt.tight_layout()

plt.savefig(output_path)
