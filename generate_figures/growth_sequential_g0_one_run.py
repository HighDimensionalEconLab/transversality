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
plot_name = "growth_sequential_g0_one_run"
output_path = output_dir + "/" + plot_name + ".pdf"
tag = "growth_sequential_g0_one_run"
api = wandb.Api()
project = "highdimensionaleconlab/deep_learning_transversality"
params = plot_params((10, 4.5))

# artifact and config
results = get_results_by_tag(api, project, tag, get_test_results=True, max_runs=1)
assert results.id.nunique() == 1
parameters = get_results_by_tag(api, project, tag, get_config=True, max_runs=1)
max_T_test = int(parameters["test_T"])
max_T = int(parameters["train_t_max"]) + 1


k_sol = results["k_t_sol"]
c_sol = results["c_t_sol"]
k_0 = results["k_t_approx"]
c_0 = results["c_t_approx"]
k_0_error = results["k_rel_error"]
c_0_error = results["c_rel_error"]

t = np.array(range(0, max_T_test))


plt.rcParams.update(params)

ax_ck = plt.subplot(121)
plt.plot(k_sol, "black", linestyle="dashed", label=r"$k(t)$")
plt.plot(k_0, "black", label=r"$\hat{k}(t)$")
plt.plot(c_sol, "blue", linestyle="dashed", label=r"$c(t)$")
plt.plot(c_0, "blue", label=r"$\hat{c}(t)$")
plt.axvline(x=max_T - 1, color="0.0", linestyle="dashed")
ylim_min = 0.5 * np.amin(np.minimum(c_sol, c_0))
ylim_max = 1.2 * np.amax(np.maximum(k_sol, k_0))
plt.ylim([ylim_min, ylim_max])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), prop={"size": params["font.size"]})
plt.title(r"Capital and Consumption: $\hat{k}(t)$ and $\hat{c}(t)$")
plt.xlabel(r"Time($t$)")
plt.legend(loc="lower right")


ax_errors = plt.subplot(122, sharex=ax_ck)
plt.plot(t, k_0_error, "black", label=r"$\varepsilon_k(t)$")
plt.plot(t, c_0_error, "blue", label=r"$\varepsilon_c(t)$")
plt.title(r"Relative errors: $\varepsilon_k(t)$ and $\varepsilon_c(t)$")
plt.axvline(x=max_T - 1, color="0.0", linestyle="dashed")
plt.xlabel(r"Time($t$)")
plt.tight_layout()


time_window = [18, 23]
ave_value = 0.5 * (k_sol[time_window[0]] + k_sol[time_window[1]])
window_width = 0.015 * ave_value
matplotlib.rcParams.update({"ytick.labelsize": 8})

axins = zoomed_inset_axes(
    ax_ck,
    4,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.7, -0.3, 0.3),
    bbox_transform=ax_ck.transAxes,
)
axins.plot(k_sol, "black", linestyle="dashed")
axins.plot(k_0, "black")

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
mark_inset(ax_ck, axins, loc1=1, loc2=3, linewidth="0.7", ls="--", ec="0.5")

plt.savefig(output_path)
