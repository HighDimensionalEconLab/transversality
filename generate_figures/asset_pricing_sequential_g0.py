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
plot_name = "asset_pricing_sequential_g0"
output_path = output_dir + "/" + plot_name + ".pdf"
tag = "asset_pricing_sequential_g0_one_run"
api = wandb.Api()
project = "highdimensionaleconlab/deep_learning_transversality"
params = plot_params((10, 4.5), fontsize=12)

# artifact and config
results = get_results_by_tag(api, project, tag, get_test_results=True)
assert results.id.nunique() == 1
parameters = get_results_by_tag(api, project, tag, get_config=True)
max_T_test = int(parameters["test_T"])
max_T = int(parameters["train_t_max"])

p_f = results["p_f_t"]
p_hat = results["p_t"]
p_error = results["p_rel_error"]
t = np.array(range(0, max_T_test))

# plotting

plt.rcParams.update(params)

ax_prices = plt.subplot(121)
plt.plot(p_f, "black", linestyle="dashed", label=r"$p_f(t)$")
plt.plot(p_hat, "black", label=r"$\hat{p}(t)$")
plt.axvline(x=max_T, color="0.0", linestyle="dashed")
ylim_min = 0.9 * np.amin(np.minimum(p_f, p_hat))
ylim_max = 1.1 * np.amax(np.maximum(p_f, p_hat))
plt.ylim([ylim_min, ylim_max])
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), prop={"size": params["font.size"]})
plt.title(r"Prices: $\hat{p}(t)$")
plt.xlabel(r"Time($t$)")

ax_errors = plt.subplot(122, sharex=ax_prices)
plt.plot(t, p_error, "black")

plt.axvline(x=max_T, color="0.0", linestyle="dashed")
plt.title(r"Relative errors: $\varepsilon_p(t)$")
plt.xlabel(r"Time($t$)")
plt.tight_layout()


time_window = [42, 47]
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
axins.plot(p_hat, "black")

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

plt.savefig(output_path)
