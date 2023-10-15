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

output_dir = "./figures"
plot_name = "growth_sequential_multiple_steady_states_four_initial_k_0"
output_path = output_dir + "/" + plot_name + ".pdf"
api = wandb.Api()
project = "highdimensionaleconlab/deep_learning_transversality"
params = plot_params((10, 5))
tag = "growth_multiple_steady_states_four_initial_k_0"
results = get_results_by_tag(api, project, tag, get_test_results=True)
parameters = get_results_by_tag(api, project, tag, get_config=True)

df = results.groupby("id")["k_t_approx"].mean()


low = results.groupby("id")["k_ss_low_norm"].mean()[0]
high = results.groupby("id")["k_ss_high_norm"].mean()[0]


k_point5 = results[results["id"] == str(df[df < low].reset_index()["id"][0])]
k_1 = results[
    results["id"] == str(df[(df < high - 2) & (df > low)].reset_index()["id"][0])
]
k_3 = results[
    results["id"] == str(df[(df < high) & (df > low + 2)].reset_index()["id"][0])
]
k_4 = results[results["id"] == str(df[df > high].reset_index()["id"][0])]

max_T_test = int(parameters["test_T"][0])
max_T = int(parameters["train_t_max"][0]) + 1
t = np.array(range(0, max_T_test))

plt.rcParams.update(params)

ax_capital = plt.subplot(121)
plt.plot(t, k_point5["k_t_approx"], "blue", label=r"$\hat{k}(t) : k_0 = 0.50$")
plt.plot(t, k_1["k_t_approx"], "gray", label=r"$\hat{k}(t) : k_0 = 1.00$")
plt.plot(t, k_3["k_t_approx"], "red", label=r"$\hat{k}(t) : k_0 = 3.00$")
plt.plot(t, k_4["k_t_approx"], "lightblue", label=r"$\hat{k}(t) : k_0 = 4.00$")

plt.plot(t, k_1["k_ss_low_norm"], "black", linestyle="dashed", label="$k_1^*$")
plt.plot(t, k_1["k_ss_high_norm"], "black", linestyle="dotted", label="$k_2^*$")

plt.axvline(x=max_T - 1, color="0.0", linestyle="dashed")
ylim_min = 0.5 * np.amin(k_point5["k_t_approx"])
ylim_max = 1.1 * np.amax(k_4["k_t_approx"])
plt.ylim([ylim_min, ylim_max])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(
    by_label.values(),
    by_label.keys(),
    prop={"size": params["font.size"]},
    loc="center right",
)
plt.title(r"Capital: $\hat{k}(t)$")
plt.xlabel(r"Time($t$)")
plt.tight_layout()

ax_consumption = plt.subplot(122, sharex=ax_capital)
plt.plot(t, k_point5["c_t_approx"], "blue", label=r"$\hat{c}(t) : k_0 = 0.50$")
plt.plot(t, k_1["c_t_approx"], "gray", label=r"$\hat{c}(t) : k_0 = 1.00$")
plt.plot(t, k_3["c_t_approx"], "red", label=r"$\hat{c}(t) : k_0 = 3.00$")
plt.plot(t, k_4["c_t_approx"], "lightblue", label=r"$\hat{c}(t) : k_0 = 4.00$")
plt.plot(t, k_4["c_ss_low_norm"], "black", linestyle="dashed", label="$c_1^*$")
plt.plot(t, k_4["c_ss_high_norm"], "black", linestyle="dotted", label="$c_2^*$")
plt.axvline(x=max_T - 1, color="0.0", linestyle="dashed")
ylim_min = 0.5 * np.amin(k_point5["c_t_approx"])
ylim_max = 1.1 * np.amax(k_4["c_t_approx"])
plt.ylim([ylim_min, ylim_max])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(
    by_label.values(),
    by_label.keys(),
    prop={"size": params["font.size"]},
    loc="center right",
)
plt.title(r"Consumption: $\hat{c}(t)$")
plt.xlabel(r"Time($t$)")
plt.tight_layout()
plt.savefig(output_path)
